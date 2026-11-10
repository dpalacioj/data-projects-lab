---
title: "06 · Ingesta de eventos (JSON → base relacional)"
parent: Proyectos
nav_order: 6
---
# 06 · Ingesta de eventos — JSON anidado a una base relacional

> Proyecto de **ingeniería de datos**. Simula un escenario de negocio real: cada
> día una tienda online te deposita un archivo JSON con pedidos anidados.
> Tienes que **validarlo**, **aplanarlo**, **cargarlo a una base relacional** y
> dejarlo listo para análisis y dashboard. Todo esto sin instalar nada pesado:
> el backend por defecto es **SQLite** (un solo archivo), pero el código está
> escrito como en producción, con credenciales en variables de entorno y SQL
> portable a Postgres cambiando solo la URL.

## Qué resuelve

**Escenario:** eres el nuevo integrante del equipo de datos. La tienda online
genera cada día un archivo JSON con todos los pedidos. Dentro de cada pedido
viene **anidado** el cliente y la **lista de items** comprados. Tu tarea:

1. Leer esos JSON del día.
2. Validarlos (los JSON del mundo real vienen rotos a veces).
3. Normalizarlos a tablas: `clientes`, `productos`, `pedidos`, `items_pedido`.
4. Cargarlos a una base de datos relacional.
5. Dar al equipo de negocio un dashboard con los KPIs típicos.

**Por qué este caso:** cubre dos cosas que los proyectos 01–05 no tocan:

- **Datos no tabulares** (JSON jerárquico, no un CSV).
- **Conexión a una base relacional** como se hace en producción (credenciales
  por entorno, URL ocultable, carga en batch, idempotencia).

**Volumen:** con los parámetros por defecto el proyecto genera y carga
**más de 1.100.000 registros** (20.000 clientes, 500 productos, 102.000
pedidos y ~978.000 items) en menos de 20 segundos. No es un juguete con 10
filas; es volumen realista.

---

## Qué aprendes aquí

| Concepto | Dónde lo ves |
|---|---|
| Leer y validar JSON anidado con Pydantic | `src/ingesta_eventos/schemas.py`, cuaderno sección 2 |
| Aplanar jerarquía JSON a tablas normalizadas | `src/ingesta_eventos/transform.py`, sección 3 |
| Credenciales fuera del código (`.env` + `python-dotenv`) | `.env.example`, `src/ingesta_eventos/config.py` |
| URL de conexión portable (SQLite ↔ Postgres) | `config.build_database_url()` |
| Ocultar password al loguear | `config.database_url_safe()` |
| DDL declarativo con SQLAlchemy Core | `src/ingesta_eventos/db.py` |
| Carga en batch e idempotencia (ON CONFLICT) | `src/ingesta_eventos/load.py` |
| Queries SQL de negocio aisladas en un módulo | `src/ingesta_eventos/queries.py` |
| Orquestar ETL extract → transform → load | `src/ingesta_eventos/pipeline.py` |
| Dashboard que solo lee de la base | `app/streamlit_app.py` |

---

## Estructura del proyecto

```
projects/06-ingesta-eventos-ecommerce/
├── README.md                        ← este archivo
├── Makefile                         ← comandos del proyecto (`make help`)
├── .env.example                     ← plantilla de variables de entorno
├── .gitignore                       ← ignora .env, data/raw/*.json, data/db/*.db
│
├── notebooks/
│   └── 01_exploracion_json.ipynb    ← recorrido paso a paso del flujo
│
├── scripts/
│   └── generar_eventos.py           ← genera JSON sintéticos (≥ 1M registros)
│
├── src/ingesta_eventos/             ← paquete Python (una pregunta → un archivo)
│   ├── __init__.py
│   ├── config.py                    ← rutas, URL de BD, parámetros, .env
│   ├── schemas.py                   ← modelos Pydantic del JSON
│   ├── db.py                        ← engine, DDL (tablas), ping, PRAGMAs
│   ├── extract.py                   ← lee y valida los JSON de disco
│   ├── transform.py                 ← aplana JSON → 4 tablas
│   ├── load.py                      ← INSERT en batch + upsert idempotente
│   ├── queries.py                   ← SQL de negocio (top productos, etc.)
│   └── pipeline.py                  ← orquesta todo (ejecutable: `python -m ...`)
│
├── app/
│   └── streamlit_app.py             ← dashboard con conteos y gráficos
│
├── tests/                           ← 21 tests, corren con SQLite en memoria
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_transform.py
│   ├── test_db.py
│   └── test_pipeline.py
│
├── data/
│   ├── raw/                         ← JSON generados (gitignored)
│   └── db/                          ← SQLite file (gitignored)
└── reports/                         ← figuras (gitignored)
```

### Qué hace cada módulo en `src/`

| Archivo | Responsabilidad única |
|---|---|
| `config.py` | Dónde están las cosas: rutas, URL de la BD (desde `.env`), parámetros del generador. Un solo sitio para cambiar. |
| `schemas.py` | Qué forma tiene cada JSON. Validación con Pydantic: si llega basura, falla acá con mensaje claro. |
| `db.py` | Cómo me conecto a la base, cómo creo las tablas, cómo hago ping. Única capa que sabe de SQLAlchemy. |
| `extract.py` | Cómo leo los JSON del disco. No transforma. |
| `transform.py` | Cómo aplano el JSON a 4 listas de dicts. **Función pura** (misma entrada → misma salida), testeable sin BD. |
| `load.py` | Cómo escribo esas listas en la base, en batch y con idempotencia. |
| `queries.py` | SQL de negocio. El dashboard solo llama funciones de aquí. |
| `pipeline.py` | Orquesta extract → transform → load. Ejecutable con `python -m ingesta_eventos.pipeline`. |

---

## Cómo correrlo (receta en 5 pasos)

### 1. Instala las dependencias

Desde la **raíz del repositorio** (no desde esta carpeta):

```bash
uv sync --extra ingesta-eventos --extra dev
```

O desde esta carpeta con el Makefile:

```bash
make setup
```

### 2. Prepara el archivo de credenciales

```bash
make env
```

Esto copia `.env.example` → `.env`. Por defecto `.env` apunta a SQLite y
funciona de inmediato; el archivo trae comentado el bloque de Postgres para
cuando quieras migrar.

> `.env` **no se commitea** (está en el `.gitignore` del repo). Es donde
> van las credenciales reales. `.env.example` sí se commitea: documenta las
> variables sin filtrar secretos. Este es el patrón estándar en producción.

### 3. Genera los datos sintéticos

```bash
make generate-data
```

Crea 60 archivos JSON en `data/raw/`, uno por día, totalizando **~1.1M
registros**. Tarda ~5 segundos.

Si tienes prisa (o estás probando):

```bash
make generate-data-small    # 7 días × 300 pedidos = ~22K registros, < 1s
```

### 4. Ejecuta el pipeline ETL

```bash
make load
```

Lee todos los JSON, valida, aplana y carga en la base. Deberías ver algo así:

```
URL de la base: sqlite:////.../data/db/eventos.db
conectado a: sqlite:////.../data/db/eventos.db  (driver: sqlite)
lotes a procesar: 60
[  1/60] eventos_...json: pedidos= 1700  items= 16455
...
[ 60/60] eventos_...json: pedidos= 1700  items= 16119

resumen de carga (filas en la base):
  clientes         19,870
  productos           500
  pedidos         102,000
  items           978,768
  TOTAL FILAS   1,101,138
tiempo total: 11.7s
```

> **Atajo:** `make pipeline` hace `generate-data` + `load` en una sola línea.

### 5. Lanza el dashboard

```bash
make serve
```

Abre `http://localhost:8501`. Verás conteos, serie temporal de ingresos,
reparto por canal y país, top productos y clientes recurrentes.

Alternativa sin UI: `make query` imprime los mismos KPIs en terminal.

---

## El patrón de producción: credenciales fuera del código

El punto didáctico central de este proyecto es que **el código se escribe
igual tanto si usas SQLite como si usas Postgres**.

Mira `src/ingesta_eventos/config.py`:

```python
def build_database_url() -> str:
    # 1) Si existe DATABASE_URL en el entorno, se usa tal cual.
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    # 2) Si hay credenciales de Postgres, arma la URL con ellas.
    if all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

    # 3) Caso contrario: SQLite local por defecto.
    return f"sqlite:///{default_sqlite}"
```

Y al loguear, nunca filtramos la password:

```python
>>> database_url_safe("postgresql+psycopg://admin:mi_clave@localhost:5432/db")
'postgresql+psycopg://admin:***@localhost:5432/db'
```

Esto es lo que una revisión de código en un equipo serio va a exigir. Si abres
un PR con una URL hardcodeada o una password en texto plano en los logs, te
lo van a devolver.

### ¿Por qué SQLite por defecto?

Porque **funciona el primer día**: no requiere instalar Postgres, no requiere
Docker, no requiere crear usuarios ni bases. Es el mismo criterio del
proyecto 05 (`load_wine` embebido). La filosofía: la primera corrida tiene
que funcionar o pierdes al estudiante.

SQLite es una base relacional de verdad: cumple SQL estándar, tiene
transacciones ACID, soporta índices y llaves foráneas. Lo único que no tiene
es un servidor separado ni control de acceso por usuario (por eso no hay
password). Para todo lo demás (aprender SQL, aprender SQLAlchemy, aprender
a modelar), es idéntica a Postgres.

### ¿Cómo migrar a Postgres sin tocar código?

1. Lanza un Postgres (por ejemplo con Docker):

   ```bash
   docker run -d --name eventos-pg \
     -e POSTGRES_USER=app_eventos \
     -e POSTGRES_PASSWORD=cambia_esto \
     -e POSTGRES_DB=eventos \
     -p 5432:5432 \
     postgres:16
   ```

2. Instala el driver:

   ```bash
   uv pip install "psycopg[binary]"
   ```

3. Edita `.env`, comenta la línea de SQLite y descomenta el bloque de Postgres.

4. Re-ejecuta `make load`. El mismo código, con la misma función `cargar_lote`,
   usa `ON CONFLICT DO NOTHING` en vez de `INSERT OR IGNORE` automáticamente
   (el dialecto lo detecta SQLAlchemy). Ni una línea cambia en el pipeline.

---

## Correr los tests

```bash
make test
```

Deberías ver:

```
tests/test_db.py ......                   [ 28%]
tests/test_pipeline.py ....               [ 47%]
tests/test_schemas.py ......              [ 76%]
tests/test_transform.py .....             [100%]

21 passed in ~4s
```

Los tests **no dependen** de que `make generate-data` o `make load` hayan
corrido antes. Levantan una base SQLite en memoria (`sqlite:///:memory:`) con
tablas limpias y verifican:

1. `test_schemas.py` — el esquema Pydantic acepta JSONs válidos y rechaza los
   rotos (canal fuera del enum, cantidad negativa, email sin arroba, etc.).
2. `test_transform.py` — el aplanado deduplica correctamente, precalcula
   totales y subtotales, y la suma de los items cuadra con el total del pedido.
3. `test_db.py` — la conexión responde, el DDL crea las 4 tablas, la URL con
   password se enmascara al loguear, y la URL se arma correctamente tanto para
   SQLite por defecto como para Postgres desde variables de entorno.
4. `test_pipeline.py` — el flujo completo extract → transform → load funciona
   end-to-end, es idempotente (correrlo 2 veces no duplica), falla con error
   claro si no hay archivos, y las queries de negocio devuelven resultados
   correctos.

---

## Cómo adaptar esta receta a otro proyecto

Esta estructura sirve para cualquier pipeline **JSON → BD relacional**.
Para adaptarla:

1. **Esquema distinto** → cambia `schemas.py`. Define los `BaseModel` de tus
   objetos. Pydantic validará todo lo demás.
2. **Más o menos tablas** → cambia `db.py` (definición de tablas) y
   `transform.py` (aplanado). Mantén el patrón "una función pura que devuelve
   un dict de listas".
3. **Otro origen** → si en vez de JSON de disco lees de una API o una cola,
   reemplaza `extract.py`. El resto del pipeline no cambia.
4. **Otro backend de BD** → solo cambia `DATABASE_URL`. Para Postgres, instala
   `psycopg[binary]`. Para MySQL, instala `pymysql` y usa
   `mysql+pymysql://...`.
5. **Más queries** → agrégalas a `queries.py`. El dashboard las toma solas si
   las llamas desde `streamlit_app.py`.

El patrón que **no debes romper**: `app/*.py` solo llama funciones de
`queries.py`. Si la app empieza a tener queries SQL inline, llévalas a
`queries.py` inmediatamente.

---

## Lo que este proyecto no cubre (y por qué)

- **Streaming** (Kafka, pub/sub). Aquí leemos archivos ya depositados en
  disco; es el patrón "batch diario" más común. El siguiente paso natural es
  reemplazar `extract.py` por un consumidor de cola.
- **Orquestación** (Airflow, Prefect, Dagster). Para un pipeline de un solo
  paso es excesivo. Cuando tengas 10 pasos y dependencias, introduce uno.
- **Modelado dimensional** (star schema). La normalización que hacemos aquí
  es la operacional (3FN). Para analítica pesada conviene una capa dimensional
  encima; queda fuera del alcance.
- **Backfill / migraciones** (Alembic). Las tablas las creamos con
  `metadata.create_all`; para un pipeline que vive años, querrás migraciones
  versionadas.

Son extensiones que se añaden cuando el problema las pide. La receta base es
ésta.

---

## Checklist antes de dar por terminado el proyecto

- [ ] `make setup` corre sin error.
- [ ] `make env` crea `.env` desde `.env.example`.
- [ ] `make generate-data` produce ≥ 60 archivos en `data/raw/`.
- [ ] `make load` carga ≥ 900.000 filas en menos de 30 segundos.
- [ ] `make test` pasa los 21 tests en menos de 10 segundos.
- [ ] `make serve` abre el dashboard y los gráficos se renderizan con datos.
- [ ] `make query` imprime conteos y top productos en terminal.
- [ ] El archivo `.env` NO está commiteado (verifícalo con `git status`).
- [ ] El archivo `data/db/eventos.db` NO está commiteado.

Si todo esto se cumple, el proyecto está sano.

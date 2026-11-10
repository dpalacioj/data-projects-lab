---
title: "14 · Bases de datos y SQLAlchemy"
parent: Tutoriales
nav_order: 15
---
# 14 · Bases de datos relacionales y SQLAlchemy

> **Ejemplo aplicado:** este tutorial cubre la **teoría**; para verlo en
> acción sobre un pipeline real que carga 1M+ registros, ve al proyecto
> [`06-ingesta-eventos-ecommerce`](../projects/06-ingesta-eventos-ecommerce/).

## Índice

1. [¿Por qué una base de datos y no un CSV?](#1-por-qué-una-base-de-datos-y-no-un-csv)
2. [SQLite vs PostgreSQL: cuándo usar cada una](#2-sqlite-vs-postgresql-cuándo-usar-cada-una)
3. [Tres formas de conectarte: sqlite3, ORM y Core](#3-tres-formas-de-conectarte-sqlite3-orm-y-core)
4. [El patrón de producción: credenciales fuera del código](#4-el-patrón-de-producción-credenciales-fuera-del-código)
5. [Carga masiva eficiente: batches](#5-carga-masiva-eficiente-batches)
6. [Idempotencia: cargar lo mismo dos veces sin duplicar](#6-idempotencia-cargar-lo-mismo-dos-veces-sin-duplicar)
7. [Cuándo migrar de SQLite a Postgres](#7-cuándo-migrar-de-sqlite-a-postgres)

---

## 1 · ¿Por qué una base de datos y no un CSV?

Un CSV funciona mientras:

- Los datos quepan en RAM.
- Solo tú los uses.
- No tengas relaciones entre entidades (clientes, pedidos, productos…).
- No necesites concurrencia (dos procesos escribiendo a la vez).

El día que incumples una de esas cuatro, necesitas una base de datos.

```mermaid
graph LR
    A[CSV] -->|crece| B{¿Cabe en RAM?}
    B -->|sí| C[Sigue sirviendo]
    B -->|no| D[BD]
    A -->|lo necesitan N personas| D
    A -->|tiene relaciones<br/>cliente↔pedidos| D
    A -->|hay concurrencia| D

    style D fill:#aaffaa
    style A fill:#ffcccc
```

**Lo que te da una BD y un CSV no:**

- **Integridad referencial** (llaves foráneas: un pedido no puede apuntar a un cliente inexistente).
- **Consultas rápidas** con índices (`WHERE fecha BETWEEN ...` en 50ms sobre millones de filas).
- **Transacciones ACID** (si el proceso se cae a la mitad de un `UPDATE`, nada queda a medias).
- **Acceso concurrente** controlado con `locks` y aislamiento.
- **SQL**, un lenguaje declarativo que cualquier analista entiende.

---

## 2 · SQLite vs PostgreSQL: cuándo usar cada una

Las dos son bases **relacionales**. Las dos hablan SQL estándar. La diferencia está en la arquitectura.

| Caracteristica | SQLite | PostgreSQL |
|---|---|---|
| Servidor | No tiene. Es una **librería** que lee un archivo `.db`. | Sí. Proceso separado (`postgres`) que escucha en `:5432`. |
| Instalación | Viene con Python. **Cero setup.** | Instalar + configurar + crear usuarios. |
| Concurrencia de escritura | Un solo escritor a la vez (con WAL, mejora). | Miles de conexiones simultáneas. |
| Autenticación | La controla el sistema de archivos (quien tiene el `.db` entra). | Usuario + password + roles + permisos por tabla. |
| Tamaño práctico | Hasta ~100 GB. Perfecto para análisis individuales. | Terabytes. Usado por empresas enormes. |
| Casos de uso | Prototipos, apps móviles/desktop, tests, análisis locales, caching. | Producción multiusuario, APIs web, data warehouses pequeños. |

**Regla práctica:**

- Estás aprendiendo o prototipando → **SQLite**.
- Es un proyecto de clase → **SQLite** (cero fricción, tu profesor lo clona y funciona).
- Lo van a usar varios procesos/personas a la vez → **Postgres**.
- Es una app en producción → **Postgres** (o MySQL, o la nube: BigQuery, Snowflake…).

**Lo bueno:** si escribes el código con **SQLAlchemy**, migrar de una a otra es cambiar una URL. Más abajo lo ves.

---

## 3 · Tres formas de conectarte: `sqlite3`, ORM y Core

### Opción A: `sqlite3` crudo (nativo de Python)

```python
import sqlite3
conn = sqlite3.connect("eventos.db")
cur = conn.cursor()
cur.execute("SELECT * FROM clientes LIMIT 5")
for row in cur.fetchall():
    print(row)
conn.close()
```

**Bueno para:** scripts simples, "voy a leer una tabla y chao".
**Malo para:** cambiar de backend (solo SQLite), proyectos grandes (SQL hardcodeado por todos lados).

### Opción B: SQLAlchemy ORM

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase): pass
class Cliente(Base):
    __tablename__ = "clientes"
    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str]

# Ahora trabajas con objetos Python, no con SQL:
cliente = session.query(Cliente).filter_by(id=1).first()
```

**Bueno para:** apps web (CRUDs con muchas relaciones), equipos que vienen de Rails/Django.
**Malo para:** ETL masivo. Instanciar 1M objetos Python es lentísimo y consume RAM a saco.

### Opción C: SQLAlchemy Core (la que usamos en el curso)

```python
from sqlalchemy import Table, Column, Integer, String, MetaData, create_engine, insert

metadata = MetaData()
clientes = Table(
    "clientes", metadata,
    Column("id", Integer, primary_key=True),
    Column("nombre", String),
)

engine = create_engine("sqlite:///eventos.db")
metadata.create_all(engine)

with engine.begin() as conn:
    conn.execute(insert(clientes), [{"id": 1, "nombre": "Ana"}, {"id": 2, "nombre": "Juan"}])
```

**Bueno para:** pipelines ETL, análisis, cualquier cosa que mueva volumen.
**Malo para:** apps con lógica de negocio compleja atada a objetos.

**Lo que más gana Core sobre ORM:** `conn.execute(insert(...), [dict1, dict2, ..., dict5000])` en una sola llamada. Para cargar 1M filas tardas segundos, no minutos.

---

## 4 · El patrón de producción: credenciales fuera del código

**Nunca** escribas esto:

```python
# MAL
engine = create_engine("postgresql://admin:mi_password@db.corp.com:5432/prod")
```

Esa password acaba de ir a git. Para siempre. Aunque la borres, queda en el historial.

**El patrón correcto** tiene tres capas:

### Capa 1: `.env` (no se commitea)

```bash
# .env — gitignored
DB_USER=app_user
DB_PASSWORD=la_password_real
DB_HOST=db.corp.com
DB_PORT=5432
DB_NAME=prod
```

### Capa 2: `.env.example` (sí se commitea)

```bash
# .env.example — commiteado como plantilla
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=5432
DB_NAME=
```

Documenta las variables sin filtrar secretos. El próximo dev clona el repo y sabe qué tiene que llenar.

### Capa 3: código que las lee

```python
import os
from dotenv import load_dotenv

load_dotenv()  # carga .env al os.environ

def construir_url():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"
```

### Y una regla extra: nunca loguees la password

```python
def url_segura(url: str) -> str:
    """Enmascara la password al imprimir la URL."""
    if "@" not in url or ":" not in url.split("://")[1].split("@")[0]:
        return url
    scheme, rest = url.split("://", 1)
    userinfo, hostinfo = rest.split("@", 1)
    user, _ = userinfo.split(":", 1)
    return f"{scheme}://{user}:***@{hostinfo}"

print(url_segura(DATABASE_URL))
# postgresql+psycopg://admin:***@db.corp.com:5432/prod
```

Si tus logs van a CloudWatch, Datadog o Grafana, la password **no puede estar en claro**. Esta función es una línea de defensa básica.

En el proyecto 06 ves este patrón funcionando en [`src/ingesta_eventos/config.py`](../projects/06-ingesta-eventos-ecommerce/src/ingesta_eventos/config.py).

---

## 5 · Carga masiva eficiente: batches

Insertar fila por fila es lentísimo. La diferencia con un batch grande es brutal:

```python
# MAL: 1 fila = 1 round-trip. Para 100.000 filas, 100.000 round-trips.
for fila in filas:
    conn.execute(insert(tabla), fila)

# BIEN: 1 llamada con 100.000 dicts. Un solo round-trip.
conn.execute(insert(tabla), filas)

# MEJOR (si son millones): batches de 5.000-10.000.
for i in range(0, len(filas), 5000):
    conn.execute(insert(tabla), filas[i:i+5000])
```

**¿Por qué batches y no un solo `execute` con todas las filas?**

- Memoria: cada batch pasa por un buffer en la BD. Un batch de 1M puede quedarse sin RAM.
- Lock: en SQLite un `INSERT` bloquea la tabla. Batches cortos liberan el lock más seguido.
- Visibilidad: si algo falla en el batch 20, los primeros 19 ya están commiteados y visibles.

**En SQLite, activa estas PRAGMAs para cargas masivas:**

```python
from sqlalchemy import event

@event.listens_for(engine, "connect")
def _pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")       # permite lecturas durante escrituras
    cur.execute("PRAGMA synchronous=NORMAL")     # fsync() menos agresivo
    cur.execute("PRAGMA foreign_keys=ON")         # habilita FK (apagadas por defecto!)
    cur.close()
```

Con esto, cargar 1M filas en SQLite tarda ~10 segundos en vez de varios minutos.

---

## 6 · Idempotencia: cargar lo mismo dos veces sin duplicar

Un pipeline ETL típico corre cada día. Si por algún motivo lo re-ejecutas (fallo, reproceso, backfill), **no deberías duplicar datos**.

**Mala estrategia:**

```python
# Si el pedido ya existe, falla con IntegrityError
conn.execute(insert(pedidos), filas)
```

**Buena estrategia (SQLite):** `INSERT OR IGNORE`

```python
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

stmt = sqlite_insert(pedidos).values(filas).prefix_with("OR IGNORE")
conn.execute(stmt)
```

**Buena estrategia (Postgres):** `ON CONFLICT DO NOTHING`

```python
from sqlalchemy.dialects.postgresql import insert as pg_insert

stmt = pg_insert(pedidos).values(filas).on_conflict_do_nothing()
conn.execute(stmt)
```

**Buena estrategia (independiente del backend):** detecta el dialecto y escoge.

```python
dialecto = engine.dialect.name
if dialecto == "sqlite":
    stmt = sqlite_insert(tabla).values(filas).prefix_with("OR IGNORE")
elif dialecto == "postgresql":
    stmt = pg_insert(tabla).values(filas).on_conflict_do_nothing()
else:
    stmt = insert(tabla).values(filas)
```

**Pro-tip:** si tu tabla es "datos del día X", en vez de `DO NOTHING` muchos pipelines borran los datos del día X antes de cargar. Es más simple que gestionar upserts complejos.

---

## 7 · Cuándo migrar de SQLite a Postgres

Si tu código usa **SQLAlchemy Core** (no `sqlite3` crudo), la migración es cambiar una variable de entorno:

```bash
# antes
DATABASE_URL=sqlite:///data/eventos.db

# después
DATABASE_URL=postgresql+psycopg://usuario:password@host:5432/db
```

**No cambias ni una línea de código.** Esa es la ventaja de usar un ORM/Core que abstrae el dialecto.

**Señales de que llegó el momento:**

- Dos procesos escribiendo a la vez empiezan a dar `database is locked`.
- Tu equipo quiere conectarse desde distintas máquinas.
- La BD se acerca a 100 GB.
- Necesitas tipos avanzados (JSONB, arrays, geolocalización, full-text search).
- Necesitas control de acceso por usuario/rol.

**Para arrancar Postgres en 30 segundos con Docker:**

```bash
docker run -d --name eventos-pg \
  -e POSTGRES_USER=app \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=eventos \
  -p 5432:5432 \
  postgres:16
```

Añades `psycopg[binary]` a tus deps, actualizas `DATABASE_URL`, re-corres el pipeline. Fin.

---

## Referencias

- Docs oficiales de SQLAlchemy: <https://docs.sqlalchemy.org/>
- SQLite PRAGMAs: <https://www.sqlite.org/pragma.html>
- The Twelve-Factor App (capítulo III, config): <https://12factor.net/config>
- Proyecto del curso donde todo esto está implementado:
  [`06-ingesta-eventos-ecommerce`](../projects/06-ingesta-eventos-ecommerce/)

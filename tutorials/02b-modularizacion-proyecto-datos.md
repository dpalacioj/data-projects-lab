---
title: "02b · Modularización de un proyecto de datos"
parent: Tutoriales
nav_order: 4.5
---
# Tutorial 02b — Modularización: ¿por qué un proyecto de datos no es un solo notebook?

Última revisión: 2026-10-31. Prerrequisito: [tutorial 02](02-configuracion-proyecto-uv.md)
(sabes qué es `pyproject.toml` y qué hace `uv sync`).

Este tutorial **no** enseña a programar nada. Enseña a **leer la forma** de un
proyecto de datos antes de abrir el primer archivo. Al terminar vas a poder
mirar una carpeta nueva y decir "esto es un `src/`, aquí van funciones; esto
es `notebooks/`, aquí se explora; esto es `data/`, aquí no se edita a mano"
sin que nadie te lo diga.

Si solo has trabajado con uno o dos notebooks sueltos, este tutorial resuelve
la pregunta que casi nadie te explica: **¿para qué tantas carpetas?**

---

## El problema que la modularización resuelve

Imagina este escenario, que es el más común en un primer proyecto de datos:

> Tienes un notebook llamado `analisis.ipynb`. Dentro: la lectura del CSV, la
> limpieza, el modelo entrenado, los gráficos, las conclusiones y, pegado al
> final, un bloque con `pickle.dump` que guarda el modelo en la misma carpeta.
> El CSV original está dos carpetas arriba. Compartes el notebook con una
> compañera. No le funciona.

Lo que falla no es tu código. Falla **la estructura**: el notebook mezcla
cuatro roles distintos como si fueran uno solo:

| Rol | Qué hace | Quién lo necesita |
|---|---|---|
| **Datos** | guardar el insumo original | quien replica el análisis |
| **Lógica reusable** | funciones de limpieza, features, métricas | quien quiere usarlas sin copiar y pegar |
| **Experimentación** | probar ideas, ver qué pasa, graficar | tú, cuando exploras |
| **Entregable** | el modelo final, el reporte, la app | quien recibe el resultado |

Si todo vive en un archivo, cualquiera que necesite **solo uno** de esos
roles tiene que leer todos los demás. Y, lo que es peor, si cambias algo en
un rol (ej. una regla de limpieza) tienes que **buscarlo** entre todo lo
demás.

**Modularizar es separar cada rol en su propia carpeta**, con la regla de
que lo que vive en una no se mezcla con lo que vive en otra. El beneficio
real no es "quedar bonito": es que **el código deja de depender de ti para
poder correr**.

---

## La metáfora de la cocina

Si todavía sientes que es ceremonia, piensa en una cocina profesional:

| Carpeta en el proyecto | Equivalente en la cocina |
|---|---|
| `data/` | **La despensa.** Ingredientes crudos. Nadie cocina ahí. |
| `src/` | **El recetario del chef.** Técnicas reusables, probadas. |
| `notebooks/` | **La mesa de pruebas.** Donde pruebas combinaciones y descartas. |
| `scripts/` | **El menú del día.** Órdenes concretas: "haz esto, ahora". |
| `models/` | **La nevera de salsas base.** Preparaciones que tardaron en salir y guardas para servir. |
| `reports/` | **Lo que llega al cliente.** El plato emplatado, no el proceso. |
| `app/` | **El restaurante.** La interfaz por donde el cliente pide. |
| `tests/` | **El protocolo de calidad.** Lo que verifica que la receta sigue dando el mismo sabor. |

Nadie confunde la despensa con el plato servido. En un proyecto de datos
pasa lo mismo: cuando mezclas despensa y plato, el proyecto deja de ser
reproducible.

---

## El mapa canónico: carpeta por carpeta

Esta es la estructura típica de un proyecto de datos como los que ves en
`projects/00-segmentacion-clientes/` dentro de este repo. No todos los
proyectos tienen todas las carpetas; aparecen conforme el proyecto las
necesita.

```
mi-proyecto/
├── pyproject.toml       # qué dependencias usa este proyecto (tutorial 02)
├── README.md            # qué es, cómo correrlo
├── Makefile             # atajos reproducibles (tutorial 03)
│
├── data/                # datos crudos y procesados (NUNCA editados a mano)
│   ├── raw/
│   └── processed/
├── notebooks/           # exploración, experimentos, "pensar en voz alta"
├── src/                 # código reusable (funciones, clases, pipelines)
│   └── mi_proyecto/
│       ├── __init__.py
│       ├── config.py
│       ├── features.py
│       └── modeling.py
├── scripts/             # ejecutables cortos de línea de comandos
├── models/              # modelos entrenados serializados
├── reports/             # entregables: PDFs, HTMLs, figuras finales
├── app/                 # interfaz de usuario (Streamlit, FastAPI, etc.)
└── tests/               # pruebas automáticas de lo que vive en src/
```

### `data/` — los datos crudos y procesados

**Qué guarda:** los archivos de insumo (CSV, Parquet, JSON) y las versiones
transformadas que son costosas de recalcular.

**Regla de oro:** `data/raw/` es **de solo lectura**. Lo que llega ahí no se
edita nunca a mano. Si necesitas cambiar algo, haces un script que
transforme `raw/` → `processed/`. Así siempre puedes volver al punto de
partida.

**Por qué no commiteas datos grandes:** git no está hecho para archivos
binarios de GB. Los datos crudos suelen ir en `.gitignore` y se documentan
con un `data/README.md` que diga de dónde se bajan.

### `notebooks/` — el laboratorio

**Qué guarda:** notebooks de **exploración**. Preguntas abiertas: "¿cómo se
distribuye esta variable?", "¿qué pasa si cambio este umbral?", "¿tiene
sentido este clúster?".

**Qué NO guarda:** lógica que vas a reutilizar. Si una función de limpieza
funciona bien en el notebook y la vas a usar en el modelo final, **se muda
a `src/`**. El notebook la importa, no la define.

**Regla práctica:** si copias y pegas una celda de un notebook a otro, esa
celda debería estar en `src/`.

### `src/` — el código reusable

**Qué guarda:** funciones, clases, pipelines. Código probado. Lo que
cualquier parte del proyecto (notebook, script, app, test) puede
`import` y usar.

**Por qué `src/mi_proyecto/` y no solo `mi_proyecto/` en la raíz:** evita
confusiones de importación. Con `src/` tu paquete solo se encuentra si
está instalado (`pip install -e .`), lo cual obliga a que todo mundo (tú
incluido) trabaje con el paquete bien instalado, no con rutas relativas
frágiles.

Este rol se cubre a fondo en el [tutorial 06](06-del-notebook-al-paquete/).

### `scripts/` — las órdenes concretas

**Qué guarda:** archivos `.py` cortos que se ejecutan desde la terminal con
`python scripts/entrenar.py`. Cada script hace **una cosa**: entrenar un
modelo, generar las figuras del reporte, limpiar un dataset.

**Diferencia con `src/`:** `src/` es la biblioteca ("sé hacer esto"),
`scripts/` es la orden ("hazlo ahora, con estos parámetros").

**Regla práctica:** un script típico es `argparse` + llamadas a funciones
de `src/` + guardar resultado. Si un script tiene más de ~100 líneas, parte
de él debería estar en `src/`.

### `models/` — artefactos entrenados

**Qué guarda:** modelos serializados (`.pkl`, `.joblib`, `.onnx`), vectorizadores
ajustados, encoders. Lo que fue costoso de obtener y quieres cargar en vez
de recalcular.

**Por qué una carpeta separada:** los modelos son **salidas**, no código.
Y suelen ser pesados: en proyectos serios van a un *registry* (MLflow, DVC)
o a almacenamiento externo, no al repo.

### `reports/` — lo que ven las personas no técnicas

**Qué guarda:** figuras finales, PDFs, HTMLs renderizados, slides. El
"producto para humanos". Lo que le pasas a tu jefa, no a otro
programador.

**Diferencia con las figuras de un notebook:** las de `reports/` ya pasaron
por una selección. No es "todo lo que exploré"; es "lo que mejor cuenta la
historia".

### `app/` — la interfaz, si existe

**Qué guarda:** la app de Streamlit, el FastAPI, el dashboard. Si el
entregable del proyecto es un modelo "en vivo", vive aquí.

**Qué NO guarda:** la lógica del modelo. La app **llama** a funciones de
`src/`. Si la lógica estuviera en la app, no podrías probarla ni usarla
desde otro lugar.

### `tests/` — la red de seguridad

**Qué guarda:** pruebas automáticas. Comprueban que las funciones de `src/`
siguen devolviendo lo que deben cuando algo cambia.

**Por qué:** sin tests, un cambio inocente en una función de limpieza puede
romper el modelo sin que nadie se entere hasta demasiado tarde. Un test es
una afirmación concreta: "si le paso esto, me devuelve esto".

### Otros nombres que verás

- `prompts/` — en proyectos con LLM, guarda los prompts como archivos de
  texto o YAML. Mismo principio que `src/`: son activos reusables.
- `docs/` — documentación del proyecto más allá del README (ADRs,
  arquitectura, guías).
- `.github/` — configuración de CI (ver [tutorial 01b](01b-tu-primer-ci.md)).
- `trabajo-final/`, `tutoriales/` — propios de un repo educativo como este.

---

## La única regla verdaderamente importante

Cada archivo tiene **un dueño** y **un tipo de dependiente**:

```
notebooks/ ─┐
scripts/   ─┼──> src/  ──> data/ (lee), models/ (escribe)
app/       ─┤
tests/     ─┘
```

- `src/` **no** importa de notebooks ni de scripts.
- Notebooks, scripts, app y tests **sí** importan de `src/`.
- Nada importa de `data/`; se lee con rutas controladas desde `src/config.py`.

Si esta regla se rompe —por ejemplo, un notebook define una función y un
script la copia— el proyecto empieza a divergir. Es la fuente #1 de "en mi
máquina sí funciona".

---

## Un ejemplo en este repo

Abre `projects/00-segmentacion-clientes/` y verás esta misma estructura
aplicada a un caso real (segmentación de clientes con RFM, K-Means y
DBSCAN sobre Online Retail II):

```
projects/00-segmentacion-clientes/
├── README.md
├── GUIA_LECTURA.md
├── DICCIONARIO_DATOS.md
├── data/            # dataset Online Retail II (raw/processed)
├── notebooks/       # exploración paso a paso, uno por bloque de la clase
├── src/             # funciones de RFM, features, clustering, scoring
├── scripts/         # p. ej. figuras_u2.py (regenera las cifras del reporte)
├── models/          # KMeans y DBSCAN entrenados
├── reports/         # entregable para quien no abre notebooks
├── app/             # app de consulta por cliente (opcional)
├── prompts/         # prompts del asistente del proyecto
├── tests/           # test_cifras_referencia.py vigila que los números no cambien
└── tutorials/       # material específico del caso
```

El `scripts/figuras_u2.py` **llama** a funciones de `src/`, lee de `data/`,
guarda en `reports/`. El notebook de clase hace exactamente lo mismo, pero
de forma interactiva. Si tienes que cambiar cómo se calcula un segmento,
cambias **una función en `src/`** y los dos recorridos se actualizan solos.

Eso es lo que la modularización compra.

---

## Síntomas de un proyecto no modularizado

Si estás trabajando en un proyecto y notas alguna de estas señales, falta
estructura:

- **"¿Dónde estaba esa función?"** — porque está copiada en tres notebooks.
- **"Funcionaba ayer."** — porque el notebook depende del orden en que se
  ejecutaron las celdas, no del código.
- **"No sé qué CSV estoy leyendo."** — porque la ruta está hardcoded en
  cada archivo distinto, en vez de centralizada en `config.py`.
- **"El modelo del compañero no carga."** — porque está guardado dentro de
  `notebooks/`, con `pickle.dump("modelo.pkl")`, en una ruta relativa que
  cambia según desde dónde ejecutes el notebook.
- **"Tocar la limpieza me rompió el gráfico."** — porque la limpieza vivía
  dentro del notebook del gráfico, no en una función aislada con tests.

Modularizar no quita estos problemas por decreto. Pero cuando aparecen, te
dice **dónde** mirar y **dónde** arreglarlo.

---

## Checklist antes de llamar "proyecto" a tu proyecto

Antes de compartir una carpeta como proyecto, pregúntate:

- [ ] ¿Un `git clone` + `uv sync` + `make <algo>` lleva al mismo resultado
      que ves tú?
- [ ] Si alguien quiere **solo entrenar** el modelo, ¿puede sin abrir un
      notebook?
- [ ] Si alguien quiere **solo ver el reporte**, ¿puede sin instalar nada?
- [ ] Si cambias **una** regla de negocio (umbral, filtro), ¿tocas un solo
      archivo?
- [ ] Si borras `notebooks/` entero, ¿el modelo y el reporte siguen
      pudiéndose regenerar?

Si la respuesta a todas es sí, tu modularización está haciendo su trabajo.

---

## Qué sigue

- Para montar el `src/` con todo (paquete instalable, tests, el patrón
  completo), ve al [tutorial 06 — Del notebook al paquete](06-del-notebook-al-paquete/).
- Para automatizar los pasos que viven en `scripts/` y `tests/` con atajos
  reproducibles, ve al [tutorial 03 — El Makefile del curso](03-el-makefile-del-curso.md).
- Para que los datos de `data/` tengan un contrato y no cambien bajo tus
  pies, ve al [tutorial 05 — Datos como código](05-datos-como-codigo/).

Vuelve al [índice de tutoriales](README.md) para ver la ruta completa.

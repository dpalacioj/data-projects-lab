---
title: Tutoriales
nav_order: 3
has_children: true
---
# Tutoriales — índice y orden de lectura

Última revisión: 2026-10-25.

Los tutoriales son material transversal. No reemplazan a las
[sesiones](../sesiones/) (que son el guión de clase), sino que las apoyan: cada
uno aísla un tema concreto para que puedas practicarlo sin la presión del
caso completo.

**La numeración de los archivos ya refleja el orden lógico de lectura.** Lee
en orden. Los números 90+ son complementos opcionales fuera de la ruta
mínima.

---

## Ruta recomendada

### Fundamentos — antes de abrir cualquier notebook

| # | Tutorial | Qué resuelve | Prerrequisito |
|---|---|---|---|
| 00 | [`00-ambientes-venv-conda-uv/`](00-ambientes-venv-conda-uv/) | Qué es un ambiente de Python, por qué `uv` y no otros | Ninguno |
| 01 | [`01-git-flujo-de-trabajo.md`](01-git-flujo-de-trabajo.md) | Clonar, rama, commit, PR, qué hace CI | Una cuenta de GitHub |
| 01b | [`01b-tu-primer-ci.md`](01b-tu-primer-ci.md) | Construye tu propio repo con CI desde cero (teoría + práctica) | Tutorial 01 |
| 01c | [`01c-tu-pagina-con-github-pages.md`](01c-tu-pagina-con-github-pages.md) | Publica tu repo como sitio web con GitHub Pages, sin crear contenido nuevo | Tutorial 01 |
| 02 | [`02-configuracion-proyecto-uv.md`](02-configuracion-proyecto-uv.md) | Cómo se estructura un proyecto con `uv` | Tutoriales 00 y 01 |
| 03 | [`03-el-makefile-del-curso.md`](03-el-makefile-del-curso.md) | Por qué un `Makefile` ahorra memoria y errores | Tutorial 02 |

### Notebooks como lo que son: experimentación

| # | Tutorial | Qué resuelve | Prerrequisito |
|---|---|---|---|
| 04 | [`04-limpieza-datos-sucios-polars/`](04-limpieza-datos-sucios-polars/) | Cómo abordar datos sucios con Polars sin pánico | Fundamentos |
| 05 | [`05-datos-como-codigo/`](05-datos-como-codigo/) | Tratar los datos como contratos (dataset card, specs) | Tutorial 04 |

### De notebook a entregable

| # | Tutorial | Qué resuelve | Prerrequisito |
|---|---|---|---|
| 06 | [`06-del-notebook-al-paquete/`](06-del-notebook-al-paquete/) | Pasar de celdas sueltas a módulos `src/` | Fundamentos |
| 12 | [`12-mlflow-y-fastapi-para-principiantes.md`](12-mlflow-y-fastapi-para-principiantes.md) | Trackear experimentos con MLflow y exponer el modelo con FastAPI | Tutorial 06 |

### Transversales — se leen cuando el tema aparece

Estos no tienen orden estricto entre sí. Son cajones de herramientas que vas
a abrir cuando una sesión o un proyecto los pida.

| # | Tutorial | Cuándo abrirlo |
|---|---|---|
| 07 | [`07-tecnicas-preprocesamiento.md`](07-tecnicas-preprocesamiento.md) | Antes de entrenar cualquier modelo tabular |
| 08 | [`08-algoritmos-de-clasificacion.md`](08-algoritmos-de-clasificacion.md) | Al elegir modelo para un problema de clasificación |
| 09 | [`09-metricas-ml-regresion-vs-clasificacion.md`](09-metricas-ml-regresion-vs-clasificacion.md) | Al decidir cómo medir el modelo |
| 10 | [`10-validacion-y-optimizacion-hiperparametros.md`](10-validacion-y-optimizacion-hiperparametros.md) | Cuando el modelo "funciona" y hay que ajustarlo bien |
| 11 | [`11-interpretabilidad-modelos-ml.md`](11-interpretabilidad-modelos-ml.md) | Cuando necesitas explicar por qué el modelo predice lo que predice |

### Complementos (90+) — opcionales, no requeridos

Fuera de la ruta mínima. Léelos si te interesa el tema.

| # | Tutorial | De qué habla |
|---|---|---|
| 90 | [`90-configuracion-inicial-claude-code.md`](90-configuracion-inicial-claude-code.md) | Cómo configurar Claude Code como asistente de desarrollo |
| 91 | [`91-introduccion-mcp.md`](91-introduccion-mcp.md) | Qué es MCP (Model Context Protocol) y para qué sirve |
| 92 | [`92-recursos-herramientas-ia.md`](92-recursos-herramientas-ia.md) | Catálogo de herramientas de IA útiles para el curso |
| 93 | [`93-fine-tuning-y-por-que-en-otra-instancia.md`](93-fine-tuning-y-por-que-en-otra-instancia.md) | Qué es fine-tuning, cuándo tiene sentido, por qué el training vive en otra máquina. Caso: Caficulbot (asistente offline con Gemma-3n) |

---

## Convenciones

### ¿Por qué unos tutoriales son archivos sueltos y otros son carpetas?

Regla simple: **si solo hay texto, es un `.md` suelto. Si hay código
ejecutable o datos que acompañan, se envuelve en una carpeta.**

- **Archivos sueltos** (p. ej. `01-git-flujo-de-trabajo.md`,
  `03-el-makefile-del-curso.md`, `07-tecnicas-preprocesamiento.md`): son
  tutoriales de solo texto. Lees y entiendes; no hay nada que correr.
- **Carpetas** (p. ej. `00-…/`, `04-…/`, `05-…/`, `06-…/`): el tutorial
  trae **archivos de apoyo** que tienen que vivir junto al `README.md`.
  Cuatro casos hoy:

  | Carpeta | Qué trae además del README |
  |---|---|
  | `00-ambientes-venv-conda-uv/` | Un notebook + `pyproject.toml`, `environment.yml`, `requirements.txt` y `uv.lock` para probar el mismo notebook en tres gestores de ambiente distintos |
  | `04-limpieza-datos-sucios-polars/` | Notebook de **ejercicio** + notebook de **solución** + CSV sintético + script que lo genera |
  | `05-datos-como-codigo/` | Un tema contado en tres `.md` temáticos + un notebook + un `taller.md` |
  | `06-del-notebook-al-paquete/` | Notebook + un **mini-paquete Python de ejemplo** (`paquete/` con `src/`, `pyproject.toml`, `tests/`) que el estudiante va construyendo |

### Otras convenciones

- Si un tutorial tiene su propia carpeta, el `README.md` interno es el
  punto de entrada.
- Cada tutorial declara su prerrequisito. Si asumes algo que no está, es
  un error del material: avisa.

---

## ¿Y las sesiones?

Las [sesiones](../sesiones/) son el guión del curso. Los tutoriales se citan
desde las sesiones cuando el tema aparece. Si estás llegando por primera vez
al repositorio, el orden es:

1. [`README.md` raíz](../README.md) → mapa general.
2. Fundamentos de esta tabla (tutoriales 00 a 03) → tienes ambiente, git y proyecto listos.
3. [`sesiones/README.md`](../sesiones/README.md) → arrancas con la sesión del día.
4. Los tutoriales de "notebooks" y "entregable" se cruzan en S02 y S03.

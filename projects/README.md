---
title: Proyectos
nav_order: 4
has_children: true
---
# Proyectos

Las carpetas tienen un prefijo numérico (`00-`, `01-`, ..., `d1-`, `d2-`...)
que marca el orden en que yo los enseño. El `ls` del disco los muestra
en ese mismo orden. No es arbitrario: va de básico a avanzado.

Hay tres grupos. Si estás empezando, lee este README completo antes de
clonar nada.

---

## Caso guía (prefijo `00-`)

Es el proyecto del curso. Las sesiones S02 y S03 trabajan sobre él.

- [`00-segmentacion-clientes/`](00-segmentacion-clientes/) — Online Retail II.
  Un año de facturas de un retailer. La pregunta: *¿quién va a volver a
  comprar en los próximos 90 días?* Recorre CRISP-DM completo (fases 1 a 5),
  con modularización más industrial (`src/segmentacion/{data, features,
  pipelines, models, evaluation}`). Si solo abres un proyecto, abre este.

---

## Didácticos completos (prefijos `01-` a `05-`)

Cuatro proyectos autocontenidos, en orden de dificultad. Cada uno recorre
un flujo entero: datos → modelo → entregable. Léelos en este orden si
estás aprendiendo.

| # | Carpeta | De qué va | Herramientas | Nivel |
|---|---|---|---|---|
| **01** | [`01-titanic/`](01-titanic/) | Primer contacto con ML. Comparar varios algoritmos sin escribir cada uno a mano | AutoML (PyCaret, FLAML), Streamlit | Básico |
| **02** | [`02-clasificador-meli/`](02-clasificador-meli/) | Clasificación supervisada en un dataset de productos de Mercado Libre | XGBoost, Pydantic | Medio |
| **03** | [`03-prediccion-precios-automoviles/`](03-prediccion-precios-automoviles/) | Regresión con modularización sencilla, tracking de experimentos y serving | MLflow, FastAPI, Gradio, Streamlit | Medio-avanzado |
| **04** | [`04-reconocimiento-imagenes/`](04-reconocimiento-imagenes/) | Fashion-MNIST con CNN. Es el ejemplo canónico del flujo "notebook paso a paso → paquete `src/` → Streamlit → FastAPI" | PyTorch, torchvision, Streamlit, FastAPI | Avanzado |
| **05** | [`05-ejemplo-completo/`](05-ejemplo-completo/) | **Plantilla del trabajo final.** Clasificador tabular (Wine) con la receta mínima: `src/` modular, `Makefile` propio, tests, `app/` con Streamlit + FastAPI, plantilla de CI | sklearn, Streamlit, FastAPI | Básico-medio |

El **patrón de modularización** que el curso quiere que entiendas se ve
mejor en el 03, el 04 y el 05: un archivo por responsabilidad (`config.py`,
`data.py`, `features.py`, `model.py`, `train.py`, `predict.py`). Nada
mágico, nada grande.

El 00 (caso guía) tiene una modularización mayor porque es un proyecto
real con subdominios. Empieza por los didácticos pequeños y vuelve al
caso guía ya calentado.

> El **05** es la plantilla específica del trabajo final. Trae además
> un `Makefile` propio, carpeta `tests/` y un `ci-ejemplo.yml` listo
> para copiar a tu propio repo. Si estás arrancando tu entregable,
> empieza clonando su estructura. Detalle de qué se espera entregar en
> [`trabajo-final/README.md`](../trabajo-final/README.md).

---

## Demos cortas (prefijo `d1-` a `d4-`)

Cada demo cubre **un** concepto en ~20 minutos. Útil cuando una sesión
toca algo de pasada y quieres el ejemplo mínimo aislado.

| ID | Carpeta | Tema | Cuándo abrirla |
|---|---|---|---|
| **d1** | [`d1-ejercicio-preprocesamiento/`](d1-ejercicio-preprocesamiento/) | Pipelines sin fuga de datos | Antes de S03 |
| **d2** | [`d2-temporal-split/`](d2-temporal-split/) | Partición temporal (no aleatoria) | Cuando los datos tienen fecha |
| **d3** | [`d3-promociones-usuarios/`](d3-promociones-usuarios/) | Construcción de la variable objetivo | Cuando el `y` no viene dado |
| **d4** | [`d4-umbrales-ejemplo/`](d4-umbrales-ejemplo/) | Umbrales y curva ROC en clasificación | Cuando el modelo "funciona pero no decide bien" |

---

## Cómo elegir uno para practicar

- Vienes del tutorial 06 ("del notebook al paquete") y quieres ver un flujo completo → **03** o **04**.
- Quieres una CNN sencilla → **04**.
- Quieres segmentar o entender clientes → **00** (caso guía).
- Quieres AutoML sin escribir modelos → **01**.
- Un concepto puntual te falló en el taller → la demo correspondiente.

---

## Convenciones

Cada proyecto declara sus dependencias como **extra** en el
`pyproject.toml` raíz. El nombre del extra NO lleva el prefijo numérico:

```bash
uv sync --extra segmentacion-clientes --extra dev    # para el 00
uv sync --extra titanic --extra dev                   # para el 01
uv sync --extra meli --extra dev                      # para el 02
uv sync --extra regresion-automoviles --extra dev     # para el 03
uv sync --extra reconocimiento-imagenes --extra dev   # para el 04
uv sync --extra ejemplo-completo --extra dev          # para el 05
```

Las demos (`d1-` a `d4-`) corren con la base del repo. No tienen extra
propio.

Datos crudos y modelos grandes no se versionan (`.gitignore` los cubre).
Cada proyecto tiene su propio script para generarlos o descargarlos.

---
title: "S04 · Aplicación y entregable"
parent: Sesiones
nav_order: 4
---
# S04 · Aplicación: del modelo al entregable

Sesión marco de trabajo autónomo (no presencial), correspondiente a la
Unidad U4.

## De qué va esta sesión

S02 y S03 te llevan hasta un modelo entrenado y guardado. Falta la pregunta
que decide si todo el trabajo valió la pena: **¿cómo lo usa alguien que no eres tú?**

CRISP-DM fase 6 (despliegue) tiene fama de ser "lo de MLOps", pero una
parte está perfectamente al alcance de este curso: envolver el modelo en
una interfaz usable. Streamlit si es para una persona de negocio que quiere
subir un archivo y ver un gráfico. FastAPI si es para otro sistema que va a
llamar al modelo cientos de veces por minuto. Gradio si quieres algo
visualmente parecido a Streamlit pero más orientado a ML interactivo.

Lo que **no** cubre este módulo:

- Despliegue a la nube (Streamlit Cloud, Hugging Face Spaces, AWS). Siguiente módulo.
- Monitoreo en producción (drift, latencia, alertas). Siguiente módulo.
- CI/CD de modelos. Siguiente módulo.

Lo que sí cubre: el salto de *"mi notebook funciona"* a *"alguien puede
clonar mi repo, correr `make setup`, y usar la app sin pedirme ayuda"*. Es
el 80% del valor.

---

## Dónde lo practicas

Tres proyectos del repositorio te muestran el patrón, en orden de
recomendación:

### 1. [`projects/04-reconocimiento-imagenes/`](../../projects/04-reconocimiento-imagenes/) — el ejemplo canónico

Es el proyecto diseñado específicamente para esta fase. Entrena una CNN
pequeña sobre Fashion-MNIST y la expone de dos formas:

- **Streamlit** ([`app/streamlit_app.py`](../../projects/04-reconocimiento-imagenes/app/streamlit_app.py)): sube una imagen, muestra top-3 clases con probabilidades.
- **FastAPI** ([`app/api_fastapi.py`](../../projects/04-reconocimiento-imagenes/app/api_fastapi.py)): `POST /predict` con la imagen, devuelve JSON.

Las dos apps importan desde el mismo [`src/reconocimiento_imagenes/predict.py`](../../projects/04-reconocimiento-imagenes/src/reconocimiento_imagenes/predict.py). No hay duplicación. Esa es la idea
central del patrón: una lógica, varias formas de invocarla.

**Empieza por aquí.** Es el más limpio y moderno.

### 2. [`projects/03-prediccion-precios-automoviles/`](../../projects/03-prediccion-precios-automoviles/) — el completo (y más viejo)

El más antiguo del repo. Tiene todas las herramientas juntas: MLflow para
tracking de experimentos, Streamlit y Gradio para interfaces, FastAPI para
API. Es útil ver cómo encajan, aunque la organización interna es más
desordenada que la del proyecto 04.

Archivos clave:

- [`app_gradio.py`](../../projects/03-prediccion-precios-automoviles/app_gradio.py) y [`app_gradio2.py`](../../projects/03-prediccion-precios-automoviles/app_gradio2.py) — dos variantes de Gradio.
- [`app_fastapi.py`](../../projects/03-prediccion-precios-automoviles/app_fastapi.py) — API con `/predict`.
- [`start_mlflow.sh`](../../projects/03-prediccion-precios-automoviles/start_mlflow.sh) — levanta MLflow local para ver experimentos.

### 3. [`projects/00-segmentacion-clientes/app/streamlit_app.py`](../../projects/00-segmentacion-clientes/app/streamlit_app.py) — el caso guía

El Streamlit del caso guía (Online Retail II). Es tu referencia para ver
cómo la Unidad 4 cierra el ciclo CRISP-DM que empezaste en U2 y U3 sobre
el mismo caso.

---

## Qué deberías poder responder al final

Preguntas de reflexión. Son las que vamos a usar en la evaluación.

1. **¿Cuándo elegir Streamlit vs. Gradio vs. FastAPI?** Cada uno existe por
   algo. Streamlit para dashboards con mucho estado; Gradio para demos de ML
   con componentes específicos (imagen, audio); FastAPI cuando lo que
   necesitas es que otro sistema llame al modelo.
2. **¿Qué pasa si cambias el modelo?** ¿La app tiene que cambiar? Si sí,
   algo está mal — la app debería leer el modelo desde un path, no
   conocerlo.
3. **¿Qué es "versión del modelo"?** ¿Por qué `metadatos.json` importa
   más que `modelo.pth` o `modelo.joblib`?
4. **¿Qué pasa si el usuario sube una imagen que no es una prenda?** El
   modelo va a predecir algo igual. ¿Cómo manejas eso?

---

## El taller final del módulo

En vez de un taller nuevo, S04 se evalúa sobre el **entregable final**:
toma el modelo que entrenaste en S03 (o entrena uno nuevo sobre Fashion-MNIST)
y envuelve una app con Streamlit **o** FastAPI. Un solo README que explique
cómo correrla. Un commit en una rama tuya. Un PR al repo con tu nombre.

Criterios:

- La app corre con `make setup && uv run streamlit run tu_app.py` (o equivalente FastAPI).
- El modelo se carga desde un archivo versionado (`.pth`, `.joblib`), no
  está hardcoded en el código.
- Si el usuario da entrada inválida, la app responde con mensaje claro, no
  con un stacktrace.
- Hay un `README.md` en tu rama que explica qué hace y cómo correr.

---

## Para antes de la evaluación

Lee el README del proyecto 04. Córrelo. Entiéndelo. Luego el 03 para
contraste. Luego ponlo en práctica sobre tu propio modelo.

Si tu Streamlit no abre en el navegador, mira primero
[`projects/04-reconocimiento-imagenes/README.md`](../../projects/04-reconocimiento-imagenes/README.md#cómo-correr-este-proyecto). El error más común es olvidar
entrenar el modelo primero.

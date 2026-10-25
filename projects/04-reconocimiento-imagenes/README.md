# Reconocimiento de imágenes — CNN sobre Fashion-MNIST

> Proyecto didáctico. Mismo patrón de los proyectos tabulares del
> repositorio, pero con una red convolucional. Entrena en CPU en pocos
> minutos.

## Qué resuelve este proyecto

**Pregunta:** dada una imagen de una prenda de ropa (28×28 en escala de
grises), ¿a qué clase pertenece?

**10 clases:** camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa,
zapatilla, bolso, bota.

**Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
(70 000 imágenes). Se descarga automático la primera vez (~30 MB).

**Modelo:** CNN simple de dos bloques convolucionales + dos capas densas.
Suficiente para ~90% de accuracy en test después de 5 épocas.

Este proyecto es el ejemplo canónico de **"flujo completo con modularización
sencilla"**:

1. Un **notebook paso a paso** que lo explica todo.
2. El mismo código, **modularizado en `src/`** (un archivo por rol: data, model, train, predict).
3. Dos entregables: **Streamlit** (interfaz para demo) y **FastAPI** (API para integrar).

---

## Estructura

```
projects/04-reconocimiento-imagenes/
├── README.md                              Este archivo
├── notebooks/
│   └── 01_cnn_paso_a_paso.ipynb           Notebook didáctico: de los datos al modelo
├── src/reconocimiento_imagenes/
│   ├── __init__.py
│   ├── config.py                          Rutas, nombres de clase, hiperparámetros
│   ├── data.py                            Descarga de Fashion-MNIST y DataLoaders
│   ├── model.py                           Arquitectura SimpleCNN
│   ├── train.py                           Bucle de entrenamiento + evaluación
│   └── predict.py                         Carga del modelo y predicción sobre una imagen
├── app/
│   ├── streamlit_app.py                   UI: sube imagen, muestra top-3
│   └── api_fastapi.py                     API: POST /predict → top-3
├── data/                                  Fashion-MNIST descargado (gitignored)
└── models/
    └── fashion_cnn.pth                    Modelo entrenado (lo generas con train)
```

---

## Cómo correr este proyecto

### 1. Instala el extra

```bash
uv sync --extra reconocimiento-imagenes --extra dev
```

Esto añade `torch`, `torchvision`, `pillow`, `streamlit`, `fastapi` y
`uvicorn`.

### 2. Lee el notebook paso a paso

Es el punto de entrada didáctico. Explica **por qué cada capa**, muestra
ejemplos de las imágenes, entrena el modelo y lo guarda:

```bash
uv run jupyter lab notebooks/01_cnn_paso_a_paso.ipynb
```

Al final del notebook, el modelo queda guardado en
`models/fashion_cnn.pth`.

### 3. Entrena desde la línea de comandos (opcional)

El mismo entrenamiento, sin notebook, usando el código ya modularizado:

```bash
uv run python -m reconocimiento_imagenes.train
```

Esto es **la misma lógica** del notebook, pero reutilizable. Es el punto
que el curso quiere que entiendas: cuando el código vive en `src/`, puedes
llamarlo desde cualquier parte (CLI, notebook, app, test).

### 4. Lanza la app Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

Abre `http://localhost:8501` y sube una imagen. La app usa el mismo
`predict.py` del paquete, no duplica código.

### 5. Lanza la API FastAPI

```bash
uv run uvicorn app.api_fastapi:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`.

Prueba desde terminal:

```bash
curl -X POST -F "file=@una_imagen.jpg" http://localhost:8000/predict
```

Respuesta:

```json
{
  "nombre_archivo": "una_imagen.jpg",
  "predicciones": [
    {"clase": "zapatilla", "probabilidad": 0.84},
    {"clase": "bota",      "probabilidad": 0.11},
    {"clase": "sandalia",  "probabilidad": 0.03}
  ]
}
```

---

## El patrón de modularización que este proyecto enseña

Un archivo por responsabilidad. Nada mágico, nada grande:

| Archivo | Rol |
|---|---|
| `config.py` | Constantes del proyecto. Un solo lugar para cambiar paths o hiperparámetros |
| `data.py` | **Qué** datos y **cómo** se cargan. Nada más |
| `model.py` | **Qué** red se usa. Solo la arquitectura, sin entrenamiento |
| `train.py` | **Cómo** se entrena y se evalúa |
| `predict.py` | **Cómo** se usa el modelo entrenado |

**Por qué esto importa.** Cuando mañana cambies el dataset, solo tocas
`data.py`. Cuando pruebes otra arquitectura, solo tocas `model.py`. El
notebook, la app Streamlit y la API FastAPI importan de **las mismas
funciones** — no hay versión "del notebook" y versión "de la app".

---

## Siguientes pasos (futuro del proyecto)

- **Transfer learning:** reemplazar `SimpleCNN` por un modelo preentrenado
  (ResNet18, MobileNetV2) sobre un dataset de más clases/tamaño.
- **Data augmentation:** flips, rotaciones, cambios de brillo en
  `data.TRANSFORM`.
- **Interpretabilidad:** Grad-CAM para ver qué regiones de la imagen activan
  cada clase.
- **Monitoreo:** registrar inferencias y detectar drift (lo que una app en
  producción necesita, no cubierto en este proyecto).

Estos pasos son material para el módulo de MLOps, no para este.

---

## Convenciones del proyecto

- Dependencias: extra `reconocimiento-imagenes` en el `pyproject.toml` raíz.
- Datos: no se versionan (están en `.gitignore`). Se descargan solo cuando
  corre `data.py`.
- Modelos: no se versionan los `.pth` grandes. Para este proyecto educativo,
  el modelo entrenado se queda localmente en `models/`.
- Semilla: `SEMILLA=42` fija los pesos iniciales para que dos corridas
  seguidas den el mismo resultado.

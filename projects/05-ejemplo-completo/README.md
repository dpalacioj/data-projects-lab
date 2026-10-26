---
title: "05 · Ejemplo completo (plantilla trabajo final)"
parent: Proyectos
nav_order: 5
---
# 05 · Ejemplo Completo — clasificador de vinos end-to-end

> Proyecto de referencia para el **trabajo final**. Muestra en pequeño lo
> que se espera de un entregable de ciencia de datos: estructura clara,
> código modular, experimentación en notebook, tests, Makefile propio,
> API y UI. Clona la receta, cambia el dataset por el tuyo.

## Qué resuelve

**Pregunta:** dada una muestra de vino con 13 mediciones químicas
(alcohol, acidez, fenoles, etc.), ¿a cuál de los 3 cultivares pertenece?

**Dataset:** `sklearn.datasets.load_wine` — 178 muestras, 3 clases, 13
features numéricas. **Viene embebido en scikit-learn:** no requiere
descarga, no requiere internet, no requiere credenciales. Elegido así a
propósito para que la receta funcione el primer día.

**Modelo:** `RandomForestClassifier` dentro de un `Pipeline` de
scikit-learn con imputación por mediana + estandarización.

Al final, el mismo modelo entrenado se usa desde tres entradas distintas
sin duplicar código:

- **CLI** (`make train`) para entrenar y guardar el `.joblib`.
- **Streamlit** (`make serve-streamlit`) para una UI de demostración.
- **FastAPI** (`make serve-api`) para una API con `POST /predict`.

---

## Estructura del proyecto

Esta es la plantilla mínima que un proyecto de ciencia de datos debería
tener. Cópiala y adapta nombres y módulos a tu caso:

```
projects/05-ejemplo-completo/
├── README.md                        ← guía del proyecto (este archivo)
├── Makefile                         ← comandos del proyecto (make help)
├── ci-ejemplo.yml                   ← plantilla de CI para tu propio repo
│
├── notebooks/
│   └── 01_experimentacion.ipynb     ← EDA, pruebas, baseline, comparación
│
├── src/ejemplo_completo/            ← paquete Python importable
│   ├── __init__.py
│   ├── config.py                    ← rutas, hiperparámetros, nombres de clase
│   ├── data.py                      ← carga del dataset
│   ├── features.py                  ← pipeline de preprocesamiento
│   ├── model.py                     ← definición del pipeline + modelo
│   ├── train.py                     ← bucle de entrenamiento + evaluación + guardado
│   └── predict.py                   ← carga del modelo entrenado + inferencia
│
├── app/                             ← interfaces para servir el modelo
│   ├── streamlit_app.py             ← UI con formulario
│   └── api_fastapi.py               ← API con POST /predict
│
├── tests/                           ← pruebas automatizadas (las corre CI)
│   ├── __init__.py
│   ├── test_data.py                 ← carga del dataset
│   ├── test_features.py             ← pipeline de preprocesamiento
│   └── test_predict.py              ← pipeline end-to-end + inferencia
│
├── models/                          ← modelo entrenado (gitignored para .joblib)
└── reports/                         ← figuras, métricas, gráficas (gitignored)
```

### Qué hace cada carpeta

| Carpeta/archivo | Rol |
|---|---|
| `README.md` | Guía para quien llega al proyecto. Explica qué resuelve, cómo correrlo, cómo adaptarlo |
| `Makefile` | Un comando por acción común (`make train`, `make test`, `make serve-api`). Si tienes que teclear más de dos palabras, va al Makefile |
| `notebooks/` | **Experimentación**, no producción. Pruebas de modelos, EDA, gráficos. El código bueno migra a `src/` |
| `src/` | Código reutilizable. Cada archivo tiene **una responsabilidad clara** |
| `app/` | Interfaces de usuario y API. Delgadas: solo importan y llaman a `src/` |
| `tests/` | Pruebas automatizadas. Se corren con `make test` y en CI |
| `models/` | Modelos entrenados serializados (`.joblib`, `.pth`). Grandes van a `.gitignore` |
| `reports/` | Resultados generados (figuras, tablas). También gitignored |
| `ci-ejemplo.yml` | Plantilla para copiar a `.github/workflows/ci.yml` en tu propio repo |

### Qué hace cada módulo en `src/`

| Archivo | Responsabilidad única |
|---|---|
| `config.py` | **Dónde están las cosas**: rutas, nombres de columnas, hiperparámetros. Un solo lugar para cambiar |
| `data.py` | **Qué datos y cómo se cargan**. No procesa, no limpia: solo carga |
| `features.py` | **Pipeline de preprocesamiento**. Imputación, escalado, codificación |
| `model.py` | **Qué modelo y cómo se arma**. Devuelve un `Pipeline` listo para `.fit()` |
| `train.py` | **Cómo se entrena y guarda**. Ejecutable con `python -m ejemplo_completo.train` |
| `predict.py` | **Cómo se usa el modelo entrenado**. Carga el `.joblib`, aplica a una muestra |

El patrón: **una pregunta → un archivo**. Si tu `train.py` también hace
preprocesamiento y define el modelo, es momento de partirlo.

---

## Cómo correrlo (receta de 5 pasos)

### 1. Instala las dependencias

Desde la **raíz del repositorio** (no desde esta carpeta):

```bash
uv sync --extra ejemplo-completo --extra dev
```

O desde esta carpeta con el Makefile:

```bash
make setup
```

### 2. Lee el notebook (opcional pero recomendado)

```bash
uv run jupyter lab notebooks/01_experimentacion.ipynb
```

Es la fase de experimentación: ver los datos, probar modelos distintos,
decidir hiperparámetros. **El notebook no es el entregable; es la
libreta de laboratorio.**

### 3. Entrena el modelo

```bash
make train
```

o directamente:

```bash
uv run python -m ejemplo_completo.train
```

Deja `models/clasificador_vino.joblib`. Debería tomar unos pocos segundos.
Verás algo como:

```
filas totales: 178
filas train:   133
filas test:    45
accuracy test: 1.0000
```

(Wine es un dataset limpio y separable; con RandomForest llega a accuracy
cercana a 1 fácilmente. En tu dataset no esperes ese número.)

### 4. Lanza la app Streamlit

```bash
make serve-streamlit
```

Abre `http://localhost:8501`. Verás un formulario con las 13 features;
ajusta valores y haz click en "Predecir cultivar".

### 5. Lanza la API FastAPI

```bash
make serve-api
```

Abre `http://localhost:8000/docs` para la documentación interactiva.
Prueba desde terminal:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.5, "malic_acid": 2.0, "ash": 2.4,
    "alcalinity_of_ash": 18.0, "magnesium": 100.0,
    "total_phenols": 2.6, "flavanoids": 2.8,
    "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.8,
    "color_intensity": 5.5, "hue": 1.0,
    "od280/od315_of_diluted_wines": 3.0, "proline": 1000.0
  }' \
  http://localhost:8000/predict
```

Respuesta:

```json
{
  "clase": "cultivar_0",
  "probabilidades": {
    "cultivar_0": 0.84,
    "cultivar_1": 0.12,
    "cultivar_2": 0.04
  }
}
```

---

## Correr los tests

```bash
make test
```

o directamente desde la raíz del repo:

```bash
uv run pytest projects/05-ejemplo-completo/tests/
```

Verás algo así:

```
tests/test_data.py ...                      [ 37%]
tests/test_features.py ..                   [ 62%]
tests/test_predict.py ...                   [100%]

8 passed in 10.85s
```

Los tests cubren tres cosas:

1. **El dataset carga** con las columnas esperadas (`test_data.py`).
2. **El preprocesador** devuelve datos escalados correctamente (`test_features.py`).
3. **El pipeline completo** entrena y predice sin errores, y el
   entrenamiento alcanza una accuracy mínima (`test_predict.py`).

**Importante:** los tests NO dependen de que `make train` haya corrido
antes. Entrenan en memoria para verificar end-to-end. Esto es
intencional: así CI los puede correr sin tener que guardar modelos.

---

## CI — Integración Continua

El repositorio del curso tiene un CI que corre los tests de este
proyecto automáticamente en cada push a `main`. Puedes verlo en
[`/.github/workflows/ci.yml`](../../.github/workflows/ci.yml) a nivel
raíz del repo.

**Para tu propio proyecto final**, usa [`ci-ejemplo.yml`](ci-ejemplo.yml)
como plantilla. Es el workflow mínimo que necesitas: clona, instala uv,
sincroniza deps, corre lint, corre tests. Cópialo a
`.github/workflows/ci.yml` en tu repo y adáptalo.

Si no sabes qué es CI o cómo se escribe un workflow, lee primero el
tutorial [`tutorials/01b-tu-primer-ci.md`](../../tutorials/01b-tu-primer-ci.md).

---

## Cómo adaptar esta receta a tu propio proyecto

Esta estructura funciona para cualquier problema tabular (regresión o
clasificación). Para adaptarla:

1. **Dataset distinto** → cambia `data.py`. Si tus datos están en un CSV,
   `Parquet` o una base, pon la carga ahí. Mantén la firma: una función
   que devuelve un `DataFrame`.
2. **Features distintas** → cambia `config.py` (`FEATURES_NUM`,
   `FEATURES_CAT`, `TARGET`) y `features.py` (añade un bloque categórico
   si tienes categoricas).
3. **Modelo distinto** → cambia `model.py`. Mantén la firma
   `construir_pipeline()` que devuelve un `Pipeline`.
4. **Entrenamiento con hiperparámetros** → añade `GridSearchCV` o lo que
   necesites dentro de `train.py`. No cambies las firmas externas.
5. **Si es regresión** → cambia `RandomForestClassifier` por
   `RandomForestRegressor`, `classification_report` por `r2_score` /
   `mean_squared_error`, y las métricas en los tests.
6. **UI distinta** → `streamlit_app.py` y `api_fastapi.py` dependen solo
   de `predict.predecir()`. Mientras esa función siga aceptando y
   devolviendo dicts, no tocas las apps.

El patrón que **no debes romper**: `app/*.py` no debe tener lógica de
negocio. Solo:

```python
from mi_paquete.predict import predecir
resultado = predecir(datos_del_usuario)
```

Si la app empieza a tener `if-else` con reglas, mueve esa lógica a `src/`.

---

## Lo que este proyecto no cubre (y por qué)

- **Imputación avanzada** (KNN, iterative). Wine no tiene nulos.
- **Codificación de categóricas**. Wine es todo numérico. Si tu
  dataset tiene categóricas, mira
  [`tutorials/07-tecnicas-preprocesamiento.md`](../../tutorials/07-tecnicas-preprocesamiento.md).
- **Interpretabilidad** (SHAP, LIME). Mira
  [`tutorials/11-interpretabilidad-modelos-ml.md`](../../tutorials/11-interpretabilidad-modelos-ml.md).
- **Tracking de experimentos** (MLflow). Mira
  [`tutorials/12-mlflow-y-fastapi-para-principiantes.md`](../../tutorials/12-mlflow-y-fastapi-para-principiantes.md).
- **Despliegue a la nube**. Siguiente módulo (MLOps).

Son extensiones. La receta de este proyecto es la base; el resto se
añade por encima cuando el problema lo pida.

---

## Checklist antes de llamarlo "terminado"

Si tu proyecto final tiene todo esto, está bien:

- [ ] Hay un `README.md` que un extraño puede leer y correr el proyecto sin pedirte ayuda.
- [ ] Hay un `Makefile` con al menos `setup`, `train`, `test`, `serve-streamlit` o `serve-api`.
- [ ] El código está en `src/`, modularizado por responsabilidad.
- [ ] Hay un notebook en `notebooks/` para la experimentación, pero NO contiene lógica que la app o la API ejecute.
- [ ] Hay al menos 3 tests que pasan: carga de datos, pipeline de features, inferencia end-to-end.
- [ ] Hay un `app/` con Streamlit o FastAPI (uno o los dos).
- [ ] Hay un `.github/workflows/ci.yml` con `make lint && make test`.
- [ ] Al abrir un PR en tu repo, aparece un check verde ✅.

Si falta algo, no es un entregable del curso; es una libreta de notas.

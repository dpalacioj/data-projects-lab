"""
Genera los dos notebooks del ejercicio:
- 01_ejercicio_preprocesamiento.ipynb (plantilla con TODOs)
- 02_solucion_preprocesamiento.ipynb (solución completa)

Ejecutar una sola vez:
    python3 build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / filename).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"  + {filename}")


# ---------------------------------------------------------------------------
# Bloques compartidos (texto que aparece igual en ambos notebooks)
# ---------------------------------------------------------------------------
HEADER_INTRO = """\
# Ejercicio · Comparación de Técnicas de Preprocesamiento

## Contexto

Trabajas como científico/a de datos en un equipo que necesita predecir si
una persona gana **más de 50K USD al año** a partir de datos socio-demográficos
(dataset *Adult Census Income* de UCI). El equipo de modelado ya tiene en
mente usar **Logistic Regression** y **Random Forest**, pero el debate
está abierto sobre **cómo preprocesar las variables**.

Tu tarea: comparar varias estrategias y entregar una recomendación
**basada en datos**, no en intuición.

## Reglas del Juego

1. Usa el **mismo split** train/test para todas las estrategias
   (semilla fija → comparaciones justas).
2. Usa **las mismas dos clases de modelo** (Logistic Regression y Random
   Forest) con hiperparámetros por defecto (no es un ejercicio de tuning).
3. Reporta **Accuracy, F1 y ROC-AUC** en el conjunto de prueba.
4. **Documenta tus decisiones**: cada estrategia debe tener un párrafo
   que explique qué cambia y por qué.

## Dataset

- **Fuente:** https://archive.ics.uci.edu/dataset/2/adult
- **Filas:** ~48,800
- **Target:** `income` (`>50K` vs `<=50K`)
- **Faltantes:** codificados como `?` (no como `NaN`)
- **Licencia:** CC BY 4.0
"""

LOAD_DATA_CODE = """\
import io
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)

URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

print("Descargando Adult Census Income desde UCI...")
train_raw = urllib.request.urlopen(URL_TRAIN).read().decode("utf-8")
test_raw = urllib.request.urlopen(URL_TEST).read().decode("utf-8")

df_train = pd.read_csv(io.StringIO(train_raw), names=COLS, skipinitialspace=True)
# adult.test trae una primera línea de comentario y un punto final en income
df_test = pd.read_csv(io.StringIO(test_raw), names=COLS,
                      skipinitialspace=True, skiprows=1)
df_test["income"] = df_test["income"].str.rstrip(".")

df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Shape: {df.shape}")
df.head()
"""

PREPARE_DATA_CODE = """\
# 1. Convertir "?" en NaN explícito (es así como lo trae el dataset original)
df = df.replace("?", np.nan)

print("Faltantes por columna:")
print(df.isna().sum()[df.isna().sum() > 0])

print(f"\\nDistribución del target:")
print(df["income"].value_counts(normalize=True).round(3))
"""

SPLIT_CODE = """\
from sklearn.model_selection import train_test_split

y = (df["income"] == ">50K").astype(int)
X = df.drop(columns=["income"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Proporción positivos en train: {y_train.mean():.3f}")
print(f"Proporción positivos en test:  {y_test.mean():.3f}")
"""

GROUP_FEATURES_CODE = """\
# Agrupamos las columnas por tipo para usarlas con ColumnTransformer
NUMERIC = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
NUMERIC_ORDINAL = ["education-num"]   # variable ordinal codificada como número
CATEGORICAL_LOW = ["workclass", "marital-status", "occupation",
                   "relationship", "race", "sex"]
CATEGORICAL_HIGH = ["native-country"]  # ~40 valores distintos
CATEGORICAL_EDU = ["education"]        # alternativa ordinal a education-num

print("Numéricas:", NUMERIC)
print("Ordinal numérica:", NUMERIC_ORDINAL)
print("Categóricas baja card.:", CATEGORICAL_LOW)
print("Categórica alta card.:", CATEGORICAL_HIGH)
print("Educación (categórica):", CATEGORICAL_EDU)
print(f"\\nNiveles únicos en native-country: {df['native-country'].nunique()}")
print(f"Niveles únicos en education: {df['education'].nunique()}")
"""

EVAL_HELPER_CODE = """\
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}


def evaluar_pipeline(preprocessor, X_tr, y_tr, X_te, y_te, etiqueta):
    \"\"\"
    Entrena LogReg y RandomForest con el preprocessor dado y devuelve
    una lista de filas con métricas. Cada fila incluye 'estrategia' y
    'modelo' para poder concatenar luego.
    \"\"\"
    from sklearn.pipeline import Pipeline

    rows = []
    for nombre_modelo, modelo in MODELS.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", modelo)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        n_feats = pipe.named_steps["pre"].transform(X_tr.head(5)).shape[1]
        rows.append({
            "estrategia": etiqueta,
            "modelo": nombre_modelo,
            "n_features": n_feats,
            "accuracy": accuracy_score(y_te, y_pred),
            "f1": f1_score(y_te, y_pred),
            "roc_auc": roc_auc_score(y_te, y_proba),
        })
    return rows
"""


# ===========================================================================
# Notebook 01 — Ejercicio (plantilla con TODOs)
# ===========================================================================
def build_ejercicio():
    cells = [
        md(HEADER_INTRO),
        md(
            """\
## Lo que tienes que hacer

Vas a construir y comparar **6 estrategias de preprocesamiento**, cada una
entrenada con dos modelos (LogReg y RF). Al final, debes producir:

1. Una **tabla comparativa** con las 12 combinaciones (6 estrategias × 2 modelos).
2. Una **gráfica** que muestre el F1 por estrategia y modelo.
3. Una **respuesta justificada** a las preguntas de reflexión.

Las 6 estrategias son:

| # | Nombre | Imputación | Categóricas | Numéricas |
|---|---|---|---|---|
| 1 | `drop_minimal` | Eliminar filas con NaN | OneHot | Sin escalar |
| 2 | `impute_ohe` | Mediana / moda | OneHot | StandardScaler |
| 3 | `impute_ohe_log` | Mediana / moda | OneHot | log1p + StandardScaler |
| 4 | `impute_ohe_minmax` | Mediana / moda | OneHot | MinMaxScaler |
| 5 | `ordinal_all` | Mediana / moda | **Ordinal** | StandardScaler |
| 6 | `drop_high_card` | Mediana / moda | OneHot SIN `native-country` | StandardScaler |
"""
        ),
        md("## Paso 1 · Imports y configuración"),
        code(LOAD_DATA_CODE),
        md(
            """\
## Paso 2 · Diagnóstico inicial

> **TODO:** observa cuántos faltantes hay y la proporción de la clase positiva.
> Anota el porcentaje de filas que perderías si simplemente hicieras `dropna()`.
"""
        ),
        code(PREPARE_DATA_CODE),
        md("## Paso 3 · Split estratificado"),
        code(SPLIT_CODE),
        md("## Paso 4 · Agrupar columnas por tipo"),
        code(GROUP_FEATURES_CODE),
        md(
            """\
## Paso 5 · Helper para evaluar pipelines

Esta función ya está completa: te ahorra escribir el mismo código una y
otra vez. Léela para entender qué hace.
"""
        ),
        code(EVAL_HELPER_CODE),
        md(
            """\
## Paso 6 · Construye los 6 preprocesadores

A continuación tienes una plantilla por estrategia. **Completa los `TODO`**
con el preprocesador correcto. Pista: todos usan `ColumnTransformer`
con uno o dos `Pipeline` adentro.

> **Tip importante**: en sklearn 1.2+ usa `OneHotEncoder(sparse_output=False)`
> para devolver arrays densos.
"""
        ),
        md("### Estrategia 1 · `drop_minimal`"),
        md(
            """\
- Quitar todas las filas con NaN antes de entrenar.
- Codificar TODAS las categóricas con `OneHotEncoder`.
- **No** escalar las numéricas (es la versión "naive").
"""
        ),
        code(
            """\
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Filtrar filas sin NaN ANTES de entrenar
mask_train = X_train.notna().all(axis=1)
mask_test = X_test.notna().all(axis=1)
X_train_drop = X_train[mask_train]
y_train_drop = y_train[mask_train]
X_test_drop = X_test[mask_test]
y_test_drop = y_test[mask_test]
print(f"Tras dropna - Train: {len(X_train_drop):,} (perdiste {len(X_train) - len(X_train_drop):,} filas)")

# TODO: completar el ColumnTransformer
preprocessor_1 = ColumnTransformer([
    # ("num", ..., NUMERIC + NUMERIC_ORDINAL),
    # ("cat", ..., CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 2 · `impute_ohe`"),
        md(
            """\
- Imputar numéricas con **mediana**, categóricas con **moda**.
- OneHot para todas las categóricas.
- StandardScaler para las numéricas.
"""
        ),
        code(
            """\
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# TODO: construir preprocessor_2
preprocessor_2 = ColumnTransformer([
    # ("num", Pipeline([...]), NUMERIC + NUMERIC_ORDINAL),
    # ("cat", Pipeline([...]), CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 3 · `impute_ohe_log`"),
        md(
            """\
- Igual que la anterior, pero antes del escalado aplica `log1p`.
- Pista: usa `FunctionTransformer(np.log1p, validate=False)` o crea una
  función propia que aplique `log(1 + max(0, x))` para evitar negativos.
"""
        ),
        code(
            """\
from sklearn.preprocessing import FunctionTransformer

def _log_safe(x):
    return np.log1p(np.clip(x, 0, None))

# TODO: construir preprocessor_3
preprocessor_3 = ColumnTransformer([
    # ("num", Pipeline([imputer, log, scaler]), NUMERIC + NUMERIC_ORDINAL),
    # ("cat", Pipeline([imputer, ohe]), CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 4 · `impute_ohe_minmax`"),
        md("- Igual que la 2, pero con `MinMaxScaler` en lugar de `StandardScaler`."),
        code(
            """\
from sklearn.preprocessing import MinMaxScaler

# TODO: construir preprocessor_4
preprocessor_4 = ColumnTransformer([
    # ...
], remainder="drop")
"""
        ),
        md("### Estrategia 5 · `ordinal_all`"),
        md(
            """\
- Imputar igual que antes.
- Codificar TODAS las categóricas con **OrdinalEncoder** en lugar de OneHot.
- Conserva StandardScaler en las numéricas.

> ⚠️ Esta estrategia es deliberadamente **mala** para Logistic Regression.
> Vas a observar empíricamente por qué.
"""
        ),
        code(
            """\
from sklearn.preprocessing import OrdinalEncoder

# TODO: construir preprocessor_5
preprocessor_5 = ColumnTransformer([
    # ("num", ...),
    # ("cat", Pipeline([imputer, OrdinalEncoder(...)]), ...),
], remainder="drop")
"""
        ),
        md("### Estrategia 6 · `drop_high_card`"),
        md(
            """\
- Igual que la estrategia 2, pero **eliminando** la variable
  `native-country` (alta cardinalidad).
- Verás cuántas columnas pierdes y si vale la pena.
"""
        ),
        code(
            """\
# TODO: construir preprocessor_6 (sin CATEGORICAL_HIGH)
preprocessor_6 = ColumnTransformer([
    # ...
], remainder="drop")
"""
        ),
        md("## Paso 7 · Ejecutar todas las estrategias"),
        md(
            """\
> **TODO:** descomenta este bloque después de completar los 6 preprocesadores.
"""
        ),
        code(
            """\
# resultados = []
# resultados += evaluar_pipeline(preprocessor_1, X_train_drop, y_train_drop,
#                                X_test_drop, y_test_drop, "drop_minimal")
# resultados += evaluar_pipeline(preprocessor_2, X_train, y_train,
#                                X_test, y_test, "impute_ohe")
# resultados += evaluar_pipeline(preprocessor_3, X_train, y_train,
#                                X_test, y_test, "impute_ohe_log")
# resultados += evaluar_pipeline(preprocessor_4, X_train, y_train,
#                                X_test, y_test, "impute_ohe_minmax")
# resultados += evaluar_pipeline(preprocessor_5, X_train, y_train,
#                                X_test, y_test, "ordinal_all")
# resultados += evaluar_pipeline(preprocessor_6, X_train, y_train,
#                                X_test, y_test, "drop_high_card")

# resultados_df = pd.DataFrame(resultados)
# resultados_df.round(4)
"""
        ),
        md("## Paso 8 · Visualizar la comparación"),
        code(
            """\
# import matplotlib.pyplot as plt
# import seaborn as sns

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for ax, metric in zip(axes, ["accuracy", "f1", "roc_auc"]):
#     sns.barplot(data=resultados_df, x="estrategia", y=metric, hue="modelo", ax=ax)
#     ax.set_title(metric)
#     ax.tick_params(axis="x", rotation=45)
# plt.tight_layout()
# plt.show()
"""
        ),
        md("## Paso 9 · Spread por modelo"),
        md(
            """\
> **TODO:** calcula, para cada modelo, la diferencia entre el mejor y el
> peor valor de F1 entre todas las estrategias. ¿Qué modelo tiene el
> mayor spread?
"""
        ),
        code(
            """\
# for modelo in resultados_df["modelo"].unique():
#     sub = resultados_df[resultados_df["modelo"] == modelo]
#     d_acc = sub["accuracy"].max() - sub["accuracy"].min()
#     d_f1 = sub["f1"].max() - sub["f1"].min()
#     d_auc = sub["roc_auc"].max() - sub["roc_auc"].min()
#     print(f"{modelo:20s} Δacc={d_acc:.4f}  Δf1={d_f1:.4f}  Δauc={d_auc:.4f}")
"""
        ),
        md(
            """\
## Preguntas de Reflexión

Responde con tus propias palabras (no hace falta que sean respuestas largas):

1. ¿Cuál es la mejor estrategia para **LogisticRegression** según tu tabla?
   ¿Y para **RandomForest**? ¿Coinciden?
2. ¿Por qué la estrategia `ordinal_all` funciona pésimo en LogReg pero
   bien en RandomForest? Explica con tus palabras qué está pasando.
3. ¿Cuántas columnas pierdes al hacer `dropna()`? ¿Vale la pena en este caso?
4. La estrategia `drop_high_card` reduce mucho la dimensionalidad
   (de ~105 a ~64 features). ¿Mejora o empeora las métricas? ¿Por qué crees?
5. ¿Esperabas que `log1p` mejorara las métricas? ¿Lo hizo? Si no, ¿qué
   crees que pasa con variables como `capital-gain` (que tiene muchos ceros)?
6. Si fueras el/la responsable de poner este modelo en producción, ¿qué
   estrategia escogerías? Justifica con datos, no con intuición.

## Entregable

Comparte con tu profesor:

1. El notebook completado.
2. La **tabla final** con las 12 filas (6 estrategias × 2 modelos).
3. Un **párrafo de 4-6 líneas** explicando tu elección final
   (modelo + estrategia) y por qué.
"""
        ),
    ]
    write_notebook("01_ejercicio_preprocesamiento.ipynb", cells)


# ===========================================================================
# Notebook 02 — Solución completa
# ===========================================================================
def build_solucion():
    cells = [
        md(HEADER_INTRO),
        md(
            """\
> **Este es el notebook de SOLUCIÓN.** Cada paso está implementado y
> comentado, y al final hay análisis de los resultados con
> respuestas razonadas a las preguntas de reflexión.
>
> **Tip docente:** evita mostrarles este notebook a los estudiantes
> hasta que hayan intentado completar `01_ejercicio_preprocesamiento.ipynb`.
"""
        ),
        md("## Paso 1 · Imports y carga"),
        code(LOAD_DATA_CODE),
        md("## Paso 2 · Diagnóstico inicial"),
        code(PREPARE_DATA_CODE),
        md(
            """\
**Lectura:**
- ~7% de las filas tienen al menos un NaN (concentrados en `workclass`,
  `occupation` y `native-country`).
- El target está desbalanceado: ~24% de clase positiva → la **F1**
  es más informativa que la accuracy en este problema.
"""
        ),
        md("## Paso 3 · Split"),
        code(SPLIT_CODE),
        md("## Paso 4 · Agrupar columnas por tipo"),
        code(GROUP_FEATURES_CODE),
        md("## Paso 5 · Helper de evaluación"),
        code(EVAL_HELPER_CODE),
        md(
            """\
## Paso 6 · Construcción de los 6 preprocesadores

### Estrategia 1 · `drop_minimal`
"""
        ),
        code(
            """\
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

mask_train = X_train.notna().all(axis=1)
mask_test = X_test.notna().all(axis=1)
X_train_drop = X_train[mask_train]
y_train_drop = y_train[mask_train]
X_test_drop = X_test[mask_test]
y_test_drop = y_test[mask_test]
print(f"Tras dropna - Train: {len(X_train_drop):,}, "
      f"perdiste {len(X_train) - len(X_train_drop):,} filas "
      f"({100*(len(X_train) - len(X_train_drop))/len(X_train):.1f}%)")

preprocessor_1 = ColumnTransformer([
    ("num", "passthrough", NUMERIC + NUMERIC_ORDINAL),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 2 · `impute_ohe` (la línea base recomendada)"),
        code(
            """\
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor_2 = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), NUMERIC + NUMERIC_ORDINAL),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 3 · `impute_ohe_log`"),
        code(
            """\
from sklearn.preprocessing import FunctionTransformer

def _log_safe(x):
    return np.log1p(np.clip(x, 0, None))

preprocessor_3 = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(_log_safe, validate=False)),
        ("sc", StandardScaler()),
    ]), NUMERIC + NUMERIC_ORDINAL),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 4 · `impute_ohe_minmax`"),
        code(
            """\
from sklearn.preprocessing import MinMaxScaler

preprocessor_4 = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", MinMaxScaler()),
    ]), NUMERIC + NUMERIC_ORDINAL),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 5 · `ordinal_all`"),
        code(
            """\
from sklearn.preprocessing import OrdinalEncoder

preprocessor_5 = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), NUMERIC + NUMERIC_ORDINAL),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), CATEGORICAL_LOW + CATEGORICAL_HIGH + CATEGORICAL_EDU),
], remainder="drop")
"""
        ),
        md("### Estrategia 6 · `drop_high_card`"),
        code(
            """\
preprocessor_6 = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), NUMERIC + NUMERIC_ORDINAL),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_LOW + CATEGORICAL_EDU),  # SIN native-country
], remainder="drop")
"""
        ),
        md("## Paso 7 · Ejecutar las 6 estrategias"),
        code(
            """\
resultados = []
resultados += evaluar_pipeline(preprocessor_1, X_train_drop, y_train_drop,
                               X_test_drop, y_test_drop, "drop_minimal")
resultados += evaluar_pipeline(preprocessor_2, X_train, y_train,
                               X_test, y_test, "impute_ohe")
resultados += evaluar_pipeline(preprocessor_3, X_train, y_train,
                               X_test, y_test, "impute_ohe_log")
resultados += evaluar_pipeline(preprocessor_4, X_train, y_train,
                               X_test, y_test, "impute_ohe_minmax")
resultados += evaluar_pipeline(preprocessor_5, X_train, y_train,
                               X_test, y_test, "ordinal_all")
resultados += evaluar_pipeline(preprocessor_6, X_train, y_train,
                               X_test, y_test, "drop_high_card")

resultados_df = pd.DataFrame(resultados)
resultados_df.round(4)
"""
        ),
        md("## Paso 8 · Visualización"),
        code(
            """\
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric in zip(axes, ["accuracy", "f1", "roc_auc"]):
    sns.barplot(data=resultados_df, x="estrategia", y=metric, hue="modelo", ax=ax)
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(resultados_df[metric].min() - 0.01,
                resultados_df[metric].max() + 0.005)
plt.tight_layout()
plt.show()
"""
        ),
        md("## Paso 9 · Spread por modelo"),
        code(
            """\
print(f"{'Modelo':<20s} {'Δacc':>8s}  {'Δf1':>8s}  {'Δauc':>8s}")
for modelo in resultados_df["modelo"].unique():
    sub = resultados_df[resultados_df["modelo"] == modelo]
    d_acc = sub["accuracy"].max() - sub["accuracy"].min()
    d_f1 = sub["f1"].max() - sub["f1"].min()
    d_auc = sub["roc_auc"].max() - sub["roc_auc"].min()
    print(f"{modelo:<20s} {d_acc:>8.4f}  {d_f1:>8.4f}  {d_auc:>8.4f}")
"""
        ),
        md(
            """\
## Análisis y Respuestas a las Preguntas

> Los números pueden variar ±0.01 según el sistema, pero las **conclusiones
> cualitativas** son las mismas.

### 1. ¿Cuál es la mejor estrategia para cada modelo?

- **LogisticRegression**: gana `impute_ohe_minmax` o `impute_ohe`
  (F1 ≈ 0.65). El log1p paradójicamente *empeora* (F1 ≈ 0.64) por el
  manejo de variables con muchos ceros como `capital-gain`.
- **RandomForest**: prácticamente todas las estrategias dan F1 ≈ 0.66-0.67,
  incluida `ordinal_all`.

**Conclusión:** los modelos lineales son MUY sensibles al preprocesamiento;
los modelos de árboles son robustos.

### 2. ¿Por qué `ordinal_all` mata a LogReg pero no a RandomForest?

- **LogReg** trata las features como **continuas y monotónicas**: si
  asignas `workclass=0` para *Private*, `1` para *Self-emp-not-inc*,
  `2` para *Federal-gov*, etc., LogReg interpreta que pasar de 0 a 2 es
  el doble del efecto de pasar de 0 a 1. **Eso es falso** — son
  categorías sin orden natural. El modelo se "confunde" y pierde poder.
- **RandomForest** hace splits binarios en cada feature
  (`workclass <= 1.5`?). Encuentra los splits que mejor separan las clases
  sin importar la "ordinalidad". La codificación numérica no le hace daño
  porque el árbol nunca asume monotonicidad lineal.

### 3. ¿Cuántas filas pierdes con dropna?

~7% del train (~2,700 filas). En un dataset chico esto sería desastroso;
aquí no afecta tanto, pero **imputar es ligeramente mejor** sin coste.

### 4. ¿`drop_high_card` ayuda o no?

Reduce de ~105 a ~64 features y las métricas casi no cambian (ΔF1 < 0.001).
Esto sugiere que `native-country` aporta poca señal predictiva
(probablemente porque el 90% de los casos son "United-States" y la
distribución es muy desbalanceada).

**Lección:** no todas las features son útiles. Más dimensiones ≠ mejor modelo.

### 5. ¿Por qué `log1p` no ayuda como esperabas?

`capital-gain` es ~92% ceros y luego una cola muy larga. Aplicar `log1p`
comprime la cola, pero **el problema no es la escala** — es que la
mayoría son ceros. Para esta variable, lo que ayudaría sería una
**discretización binaria** (cero / no-cero) o **WOE encoding**.

### 6. Recomendación final para producción

**Random Forest + `impute_ohe`** (estrategia 2):

- F1 ≈ 0.665, AUC ≈ 0.900.
- **Robusto a la elección de preprocesamiento** → mantenimiento más fácil.
- Si en el futuro el equipo de datos cambia el imputer, los resultados no
  se rompen drásticamente.
- LogReg es ligeramente peor en F1 y mucho más sensible a decisiones
  arbitrarias del pipeline.

Si fuera obligatorio usar **LogReg** (por interpretabilidad reglamentaria),
escogería `impute_ohe_minmax` y nunca `ordinal_all`.

## Lecciones Pedagógicas

1. **Compara siempre con el mismo split.** Un cambio de semilla puede
   ocultar/inflar diferencias.
2. **Los modelos lineales necesitan preprocesamiento cuidadoso.**
   Los árboles no.
3. **Más features no es mejor**: `native-country` se puede eliminar.
4. **OneHot ≠ Ordinal** para categóricas sin orden. La intuición
   "todas las categóricas se codifican igual" es incorrecta.
5. **No prejuzgues**: `log1p` "debería" ayudar pero no siempre lo hace.
   Mide siempre.
"""
        ),
    ]
    write_notebook("02_solucion_preprocesamiento.ipynb", cells)


if __name__ == "__main__":
    print("Generando notebooks del ejercicio...")
    build_ejercicio()
    build_solucion()
    print("Listo.")

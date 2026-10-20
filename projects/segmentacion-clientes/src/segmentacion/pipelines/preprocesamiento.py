"""`ColumnTransformer` para la cohorte de recompra (U3).

Firma citada por `SPEC_procesamiento_U3.md` §4:

    construir_preprocesador(num_log=NUM_LOG, num_std=NUM_STD, cat=CAT) -> ColumnTransformer

Reglas (Tabla 8 del OVA): todo `fit` ocurre dentro del `Pipeline`, despues de
la particion. Las medias que aprende el `StandardScaler` son las de la
cohorte A, no las de A ∪ B (lo verifica `tests/pipelines/test_sin_fuga.py`).

El modulo recibe pandas (la frontera Polars -> sklearn: ver §4 "Polars hasta
la matriz de variables; scikit-learn desde ahi").
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Columnas con cola larga: imputacion por mediana -> log1p -> escalado.
NUM_LOG: list[str] = ["monetario", "frecuencia", "productos_distintos"]
# Columnas numericas sin cola larga: imputacion por mediana -> escalado.
NUM_STD: list[str] = ["recencia", "antiguedad", "tasa_devolucion"]
# Columnas categoricas: `handle_unknown='ignore'` para no explotar en inferencia
# si aparece un pais no visto en entrenamiento.
CAT: list[str] = ["pais"]


def construir_preprocesador(
    num_log: list[str] | None = None,
    num_std: list[str] | None = None,
    cat: list[str] | None = None,
    incluir_prop_llm: bool = False,
) -> ColumnTransformer:
    """Devuelve el `ColumnTransformer` del caso de recompra.

    Los defaults son `NUM_LOG`, `NUM_STD`, `CAT`. Pasar listas distintas
    permite experimentar en el cuaderno 07 (ingenieria de caracteristicas)
    sin cambiar el contrato del modelo final.

    Si `incluir_prop_llm=True`, se añade un bloque `prop` con las 12
    columnas `prop_<categoria>` del modulo LLM (ver `features.texto_llm`)
    como `passthrough` (ya estan acotadas en `[0, 1]`; no se escalan).
    El cuaderno 08 usa este flag para comparar el bosque con y sin las
    columnas del LLM; cuando esta en False, las cifras de §9 no cambian.
    """
    if num_log is None:
        num_log = NUM_LOG
    if num_std is None:
        num_std = NUM_STD
    if cat is None:
        cat = CAT

    bloques = [
        (
            "log",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("sc", StandardScaler()),
                ]
            ),
            num_log,
        ),
        (
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            num_std,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ]
    if incluir_prop_llm:
        from segmentacion.features.texto_llm import CATEGORIAS_LISTA

        prop_cols = [f"prop_{c}" for c in CATEGORIAS_LISTA]
        bloques.append(("prop", "passthrough", prop_cols))
    return ColumnTransformer(bloques)

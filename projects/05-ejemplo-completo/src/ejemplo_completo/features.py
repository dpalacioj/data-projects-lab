"""Pipeline de preprocesamiento. Aisla el 'que hacer con las features' del 'que modelo usar'.

Para este dataset son todas numericas. En un proyecto con categoricas,
tambien se ensamblaria un bloque categorico con OneHotEncoder, aqui.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES_NUM


def construir_preprocesador() -> ColumnTransformer:
    """Imputacion por mediana + estandarizacion para las features numericas."""
    numerico = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [("num", numerico, FEATURES_NUM)],
        remainder="drop",
    )

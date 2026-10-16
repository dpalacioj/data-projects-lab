"""
Pipelines de preprocesamiento usando scikit-learn.

Centraliza la construcción de pipelines reutilizables para clustering.
La intuición detrás del diseño:

- Las variables RFM están muy sesgadas (long-tail). Aplicar una
  transformación logarítmica antes de escalar reduce la influencia
  desproporcionada de los outliers.
- ``StandardScaler`` deja todas las variables en la misma escala —
  imprescindible para K-Means y DBSCAN (ambos usan distancia euclidiana).
- ``ColumnTransformer`` permite tratar diferentes columnas con
  diferentes transformaciones de forma declarativa.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _log1p_transform(x: np.ndarray) -> np.ndarray:
    """Transformación logarítmica robusta a ceros (log(1+x))."""
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def build_numeric_pipeline(use_log: bool = True) -> Pipeline:
    """
    Pipeline para variables numéricas: imputación → log1p (opcional) → escalado.

    Parameters
    ----------
    use_log : bool
        Si es ``True`` aplica ``log1p`` antes del escalado. Recomendado para
        Monetary y Frequency que suelen tener distribuciones muy sesgadas.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_log:
        steps.append(("log", FunctionTransformer(_log1p_transform, validate=False)))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_categorical_pipeline() -> Pipeline:
    """
    Pipeline para variables categóricas: imputación → one-hot encoding.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessing_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    use_log: bool = True,
    n_components_pca: int | None = None,
) -> Pipeline:
    """
    Pipeline completo de preprocesamiento para clustering.

    Combina pipelines numéricos y categóricos vía ``ColumnTransformer``,
    y opcionalmente aplica PCA al final.

    Parameters
    ----------
    numeric_features : Iterable[str]
        Nombres de columnas numéricas.
    categorical_features : Iterable[str], optional
        Nombres de columnas categóricas. Si ``None``, no se usan.
    use_log : bool
        Aplicar ``log1p`` a las numéricas (recomendado para RFM).
    n_components_pca : int, optional
        Si se proporciona, agrega un paso final de PCA con ese número de
        componentes (útil para visualización 2D o para reducir colinealidad).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline listo para ``.fit_transform(X)``.
    """
    transformers = [
        ("num", build_numeric_pipeline(use_log=use_log), list(numeric_features)),
    ]
    if categorical_features:
        transformers.append(
            ("cat", build_categorical_pipeline(), list(categorical_features))
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    steps = [("preprocess", column_transformer)]
    if n_components_pca is not None:
        steps.append(("pca", PCA(n_components=n_components_pca, random_state=42)))

    return Pipeline(steps)

"""Carga del dataset Wine. Viene embebido en scikit-learn: cero descarga."""
import pandas as pd
from sklearn.datasets import load_wine

from .config import FEATURES_NUM, TARGET


def cargar() -> pd.DataFrame:
    """Devuelve el dataset como un DataFrame con columnas de features + 'cultivar'."""
    bunch = load_wine()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df[TARGET] = bunch.target
    return df


def separar_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y variable objetivo (y) sin tocar el DataFrame original."""
    X = df[FEATURES_NUM].copy()
    y = df[TARGET].copy()
    return X, y

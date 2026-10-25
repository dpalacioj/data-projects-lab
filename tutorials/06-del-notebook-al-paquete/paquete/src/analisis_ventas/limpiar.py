"""Reglas de limpieza del dataset de ventas.

Cada regla es una funcion pura: entra un DataFrame, sale un DataFrame.
No hay estado oculto, no hay orden implicito. El orden en que se aplican
las reglas vive arriba, en `limpiar`.
"""

from __future__ import annotations

import pandas as pd

from analisis_ventas.config import CANCELACION_PREFIX, PRECIO_MAX, PRECIO_MIN


def descartar_cancelaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas cuya factura empieza con el prefijo de cancelacion."""
    mascara = ~df["Invoice"].astype(str).str.startswith(CANCELACION_PREFIX)
    return df.loc[mascara].copy()


def filtrar_precio_valido(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo filas con precio en el rango valido declarado en config."""
    mascara = df["Price"].between(PRECIO_MIN, PRECIO_MAX, inclusive="both")
    return df.loc[mascara].copy()


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza en orden fijo. Esta es la unica funcion
    publica del modulo."""
    return (
        df
        .pipe(descartar_cancelaciones)
        .pipe(filtrar_precio_valido)
    )

"""Resumenes de EDA en Polars (cuartiles, cercas de Tukey, correlaciones)."""

from segmentacion.eda.resumen import (
    asociaciones,
    cercas_tukey,
    cuartiles,
    matriz_dia_hora,
    ventas_por_mes,
)

__all__ = [
    "asociaciones",
    "cercas_tukey",
    "cuartiles",
    "matriz_dia_hora",
    "ventas_por_mes",
]

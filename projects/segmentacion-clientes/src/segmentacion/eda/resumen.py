"""Resumenes de EDA en Polars para la Unidad 2.

Funciones citadas por el OVA (seccion 2.2):

- `cuartiles(df, col)`                -> dict con min/Q1/mediana/Q3/max/media.
- `cercas_tukey(df, col, k=1.5)`      -> `(inferior, superior, fraccion_fuera)`.
- `asociaciones(df, a, b)`            -> `{"pearson": ..., "spearman": ...}`.
- `ventas_por_mes(df)`                -> DataFrame por mes.
- `matriz_dia_hora(df)`               -> DataFrame largo con `(dia, hora, n)`.

Las funciones aceptan `pl.DataFrame` o `pl.LazyFrame`; donde la operacion es
agregacion, se colecta al final.
"""
from __future__ import annotations

import polars as pl

Frame = pl.DataFrame | pl.LazyFrame


def _eager(df: Frame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def cuartiles(df: Frame, col: str) -> dict[str, float]:
    """Resumen de posicion de una columna numerica."""
    s = _eager(df)[col]
    return {
        "min": float(s.min()),
        "Q1": float(s.quantile(0.25)),
        "mediana": float(s.median()),
        "Q3": float(s.quantile(0.75)),
        "max": float(s.max()),
        "media": float(s.mean()),
    }


def cercas_tukey(df: Frame, col: str, k: float = 1.5) -> tuple[float, float, float]:
    """Cercas de Tukey `(inferior, superior, fraccion_fuera)` para `col`."""
    s = _eager(df)[col]
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    riq = q3 - q1
    inferior = q1 - k * riq
    superior = q3 + k * riq
    fuera = ((s < inferior) | (s > superior)).sum()
    return inferior, superior, float(fuera) / s.len()


def asociaciones(df: Frame, a: str, b: str) -> dict[str, float]:
    """Correlaciones de Pearson y Spearman entre dos columnas."""
    e = _eager(df)
    return {
        "pearson": float(e.select(pl.corr(a, b)).item()),
        "spearman": float(e.select(pl.corr(a, b, method="spearman")).item()),
    }


def ventas_por_mes(df: Frame, col_fecha: str = "InvoiceDate", col_importe: str = "Importe") -> pl.DataFrame:
    """Agrega `col_importe` por mes truncando `col_fecha`."""
    return (
        _eager(df)
        .group_by(pl.col(col_fecha).dt.truncate("1mo").alias("mes"))
        .agg(pl.col(col_importe).sum().alias("ventas"))
        .sort("mes")
    )


def matriz_dia_hora(df: Frame, col_fecha: str = "InvoiceDate") -> pl.DataFrame:
    """DataFrame largo con conteo por `(dia, hora)` (dia 1..7, lun..dom)."""
    return (
        _eager(df)
        .group_by(
            pl.col(col_fecha).dt.weekday().alias("dia"),
            pl.col(col_fecha).dt.hour().alias("hora"),
        )
        .agg(pl.len().alias("n"))
        .sort("dia", "hora")
    )

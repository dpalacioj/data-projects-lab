"""Construccion de variables por cliente (RFM extendido) en Polars.

Firma publica:

    construir_rfm(ventas, fecha_corte: datetime) -> pl.DataFrame

Solo se usan ventas con `InvoiceDate < fecha_corte`. Columnas devueltas:
`Customer ID`, `recencia`, `antiguedad`, `frecuencia`, `monetario`,
`productos_distintos`.

En la Unidad 2 la fecha de corte es `max(InvoiceDate) + 1 dia` (snapshot). La
Unidad 3 reutiliza esta misma funcion con cortes anteriores, sin cambios de firma.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

Frame = pl.DataFrame | pl.LazyFrame


def construir_rfm(ventas: Frame, fecha_corte: datetime) -> pl.DataFrame:
    """Construye el DataFrame RFM extendido a nivel de cliente."""
    v = ventas.lazy() if isinstance(ventas, pl.DataFrame) else ventas
    v = v.filter(pl.col("InvoiceDate") < fecha_corte)

    rfm = (
        v.group_by("Customer ID")
        .agg(
            ((pl.lit(fecha_corte) - pl.col("InvoiceDate").max()).dt.total_days()).alias("recencia"),
            ((pl.lit(fecha_corte) - pl.col("InvoiceDate").min()).dt.total_days()).alias(
                "antiguedad"
            ),
            pl.col("Invoice").n_unique().alias("frecuencia"),
            pl.col("Importe").sum().alias("monetario"),
            pl.col("StockCode").n_unique().alias("productos_distintos"),
        )
        .sort("Customer ID")
    )
    return rfm.collect()

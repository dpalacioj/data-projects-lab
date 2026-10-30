"""Etiqueta supervisada `recompra` para la Unidad 3.

Firma citada por `SPEC_procesamiento_U3.md` §4:

    etiqueta_recompra(ventas, fecha_corte, horizonte_dias=90) -> pl.DataFrame

Un cliente queda etiquetado con `recompra = 1` si realiza al menos una compra
en el intervalo `[fecha_corte, fecha_corte + horizonte_dias)`. Esta ventana
define el problema de la spec de la Unidad 3 ("¿el cliente vuelve a comprar en los 90 dias
siguientes al corte?").
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

Frame = pl.DataFrame | pl.LazyFrame


def etiqueta_recompra(
    ventas: Frame,
    fecha_corte: datetime,
    horizonte_dias: int = 90,
) -> pl.DataFrame:
    """Devuelve un DataFrame con `Customer ID` y `recompra` (0/1).

    Solo contiene a los clientes que compraron en `[fecha_corte, fecha_corte + horizonte_dias)`.
    El `join` con la cohorte completa (en `cohortes.cohorte`) convierte los
    nulos en 0 para los clientes sin recompra.
    """
    v = ventas.lazy() if isinstance(ventas, pl.DataFrame) else ventas
    fin = fecha_corte + timedelta(days=horizonte_dias)
    futuro = v.filter((pl.col("InvoiceDate") >= fecha_corte) & (pl.col("InvoiceDate") < fin))
    return (
        futuro.select(pl.col("Customer ID").unique())
        .with_columns(pl.lit(1).cast(pl.Int64).alias("recompra"))
        .collect()
    )

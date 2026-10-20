"""Variable `tasa_devolucion` por cliente para la Unidad 3.

Firma citada por `SPEC_procesamiento_U3.md` §4:

    tasa_devolucion(cancelaciones, frecuencia_por_cliente, fecha_corte) -> pl.DataFrame

Hallazgo de la U2 que esta feature materializa (HALLAZGOS.md, fila 2): las
cancelaciones son un flujo paralelo con informacion de negocio. En U3 se
convierte en `n_cancelaciones_hasta_corte / frecuencia`.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

Frame = pl.DataFrame | pl.LazyFrame


def tasa_devolucion(
    cancelaciones: Frame,
    frecuencia_por_cliente: pl.DataFrame,
    fecha_corte: datetime,
) -> pl.DataFrame:
    """Devuelve un DataFrame con `Customer ID` y `tasa_devolucion`.

    - `cancelaciones` son las facturas que empiezan por `C` (ver
      `data/limpieza.separar_cancelaciones`); se filtran a `InvoiceDate < fecha_corte`.
    - `frecuencia_por_cliente` es un DataFrame con `Customer ID` y `frecuencia`.
    - Clientes sin cancelaciones previas al corte reciben `tasa_devolucion = 0.0`.
    """
    canc = cancelaciones.lazy() if isinstance(cancelaciones, pl.DataFrame) else cancelaciones
    canc_hist = canc.filter(pl.col("InvoiceDate") < fecha_corte)
    conteo = canc_hist.group_by("Customer ID").agg(
        pl.col("Invoice").n_unique().alias("n_cancelaciones")
    )
    return (
        frecuencia_por_cliente.lazy()
        .join(conteo, on="Customer ID", how="left")
        .with_columns(
            (pl.col("n_cancelaciones").fill_null(0) / pl.col("frecuencia")).alias("tasa_devolucion")
        )
        .select("Customer ID", "tasa_devolucion")
        .collect()
    )

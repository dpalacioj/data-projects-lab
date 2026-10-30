"""Reglas de limpieza para Online Retail II, en Polars.

Una funcion por regla:

- `quitar_duplicados`
- `separar_cancelaciones`  -> `(sin_cancelaciones, cancelaciones)`
- `exigir_cliente`
- `filtrar_validas`        -> Quantity > 0 y Price > 0
- `preparar_ventas`        -> encadena las cuatro con `.pipe` y agrega `Importe`

Todas aceptan `pl.DataFrame` o `pl.LazyFrame`. `preparar_ventas` registra
`n_antes/n_despues` por regla via `logging`; util en los cuadernos (02) y
en los scripts (`figuras_u2.py`).
"""

from __future__ import annotations

import logging

import polars as pl

from segmentacion.utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)

Frame = pl.DataFrame | pl.LazyFrame


def _height(df: Frame) -> int:
    """Numero de filas tanto en eager como en lazy."""
    if isinstance(df, pl.LazyFrame):
        return df.select(pl.len()).collect().item()
    return df.height


def quitar_duplicados(df: Frame) -> Frame:
    """Elimina filas duplicadas exactas (`DataFrame.unique`)."""
    return df.unique()


def separar_cancelaciones(df: Frame) -> tuple[Frame, Frame]:
    """Separa facturas que empiezan por `C` (cancelaciones) del resto."""
    es_cancelacion = pl.col("Invoice").str.starts_with("C")
    return df.filter(~es_cancelacion), df.filter(es_cancelacion)


def exigir_cliente(df: Frame) -> Frame:
    """Mantiene solo filas con `Customer ID` conocido."""
    return df.filter(pl.col("Customer ID").is_not_null())


def filtrar_validas(df: Frame) -> Frame:
    """Mantiene filas con `Quantity > 0` y `Price > 0`."""
    return df.filter(pl.col("Quantity") > 0, pl.col("Price") > 0)


def preparar_ventas(df: Frame, log: bool = True) -> tuple[Frame, Frame]:
    """Encadena las cuatro reglas y agrega `Importe = Quantity * Price`.

    Devuelve `(ventas, cancelaciones)`. Las cancelaciones se devuelven antes
    de exigir cliente y filtrar validez, para no perder las que no tienen
    `Customer ID` (relevantes para `tasa_devolucion` en U3).
    """
    n0 = _height(df)
    df = df.pipe(quitar_duplicados)
    n1 = _height(df)
    df, cancelaciones = separar_cancelaciones(df)
    n2 = _height(df)
    df = df.pipe(exigir_cliente)
    n3 = _height(df)
    df = df.pipe(filtrar_validas)
    n4 = _height(df)

    ventas = df.with_columns((pl.col("Quantity") * pl.col("Price")).alias("Importe"))

    if log:
        logger.info(
            "pipeline de limpieza: %s -> sin duplicados %s -> sin cancelaciones %s "
            "-> con cliente %s -> validas %s (%.1f%% del original)",
            f"{n0:,}".replace(",", " "),
            f"{n1:,}".replace(",", " "),
            f"{n2:,}".replace(",", " "),
            f"{n3:,}".replace(",", " "),
            f"{n4:,}".replace(",", " "),
            100 * n4 / n0 if n0 else 0.0,
        )
    return ventas, cancelaciones

"""Lectura del dataset Online Retail II con esquema declarado.

Expone:
- `ESQUEMA`: columnas que deben leerse con tipo explicito.
- `leer_excel_con_esquema(ruta)`: concatena las hojas del Excel aplicando
  los overrides de esquema.
- `leer_parquet(ruta, lazy=False)`: lectura eager o lazy del Parquet.

Los scripts y cuadernos de la Unidad 2 pasan por aqui para que la leccion
"declarar el esquema al leer" quede en un solo sitio.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ESQUEMA: dict[str, pl.DataType] = {
    "Invoice": pl.Utf8,
    "StockCode": pl.Utf8,
    "Customer ID": pl.Int64,
}


def leer_excel_con_esquema(ruta: Path) -> pl.DataFrame:
    """Lee el xlsx de Online Retail II y concatena sus dos hojas."""
    hojas = pl.read_excel(ruta, sheet_id=0, schema_overrides=ESQUEMA)
    return pl.concat(hojas.values(), how="vertical")


def leer_parquet(ruta: Path, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Lee el Parquet; `lazy=True` para los scripts que puedan beneficiarse."""
    if lazy:
        return pl.scan_parquet(ruta)
    return pl.read_parquet(ruta)

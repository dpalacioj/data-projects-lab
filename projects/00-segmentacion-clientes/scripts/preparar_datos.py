# -*- coding: utf-8 -*-
"""Prepara `data/processed/ventas_validas.parquet` y `cancelaciones.parquet`.

Es la version "comando" del cuaderno 02_limpieza: lee el raw descargado por
`scripts/descargar_datos.py`, aplica la pipeline `preparar_ventas` del
paquete (duplicados -> cancelaciones -> con cliente -> validas, mas
`Importe`) y deja los dos parquets que consumen la U2, la U3 y la app
no supervisada.

Uso (desde la carpeta del proyecto):
    uv run python scripts/preparar_datos.py

O desde la raiz del repo:
    make preparar-datos
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import polars as pl

from segmentacion.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from segmentacion.data.limpieza import preparar_ventas

RAW_PARQUET = RAW_DATA_DIR / "online_retail_II.parquet"
VENTAS_VALIDAS = PROCESSED_DATA_DIR / "ventas_validas.parquet"
CANCELACIONES = PROCESSED_DATA_DIR / "cancelaciones.parquet"


def main() -> None:
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(
            f"No encuentro {RAW_PARQUET.relative_to(PROJECT_ROOT)}. "
            "Ejecuta primero `uv run python scripts/descargar_datos.py` "
            "(o `make descargar-datos` si existe) para descargar Online Retail II."
        )

    print(f"[1/3] Leyendo raw: {RAW_PARQUET.relative_to(PROJECT_ROOT)}")
    raw = pl.read_parquet(RAW_PARQUET)

    print("[2/3] Aplicando preparar_ventas (duplicados -> cancelaciones -> cliente -> validas)...")
    ventas, cancelaciones = preparar_ventas(raw)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ventas.write_parquet(VENTAS_VALIDAS)
    cancelaciones.write_parquet(CANCELACIONES)

    print(
        f"[3/3] Guardado {VENTAS_VALIDAS.relative_to(PROJECT_ROOT)} "
        f"({ventas.height} filas) y {CANCELACIONES.relative_to(PROJECT_ROOT)} "
        f"({cancelaciones.height} filas)."
    )


if __name__ == "__main__":
    main()

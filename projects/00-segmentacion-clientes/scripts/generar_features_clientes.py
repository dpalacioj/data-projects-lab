# -*- coding: utf-8 -*-
"""Genera `data/processed/features_clientes.parquet` (snapshot RFM U2 extendido).

Es la version "comando" del paso de ingenieria de caracteristicas del flujo
no supervisado del caso guia. Lee las ventas limpias (`ventas_validas.parquet`,
producido por el notebook 02), construye la tabla por cliente con
`features.construir_features_extendidas` y la deja en el formato que
`app/streamlit_app.py` y `notebooks/10_segmentacion_no_supervisada.ipynb`
esperan.

Uso (desde la carpeta del proyecto):
    uv run python scripts/generar_features_clientes.py

O desde la raiz del repo:
    make features-segmentacion
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import polars as pl

from segmentacion.config import CLEAN_DATA_FILE, FEATURES_DATA_FILE
from segmentacion.features import construir_features_extendidas

VENTAS_VALIDAS_FILE = CLEAN_DATA_FILE.parent.parent / "processed" / "ventas_validas.parquet"


def _localizar_ventas() -> Path:
    """Prefiere `data/processed/ventas_validas.parquet` (U2); cae en `CLEAN_DATA_FILE`."""
    if VENTAS_VALIDAS_FILE.exists():
        return VENTAS_VALIDAS_FILE
    if CLEAN_DATA_FILE.exists():
        return CLEAN_DATA_FILE
    raise FileNotFoundError(
        f"No encuentro {VENTAS_VALIDAS_FILE.relative_to(PROJECT_ROOT)}. "
        "Ejecuta primero `make preparar-datos` desde la raiz del repo "
        "(o `uv run python scripts/preparar_datos.py` desde la carpeta del proyecto)."
    )


def main() -> None:
    ruta_ventas = _localizar_ventas()
    print(f"[1/3] Leyendo ventas limpias: {ruta_ventas.relative_to(PROJECT_ROOT)}")
    ventas = pl.read_parquet(ruta_ventas)

    print("[2/3] Construyendo features extendidas por cliente...")
    features = construir_features_extendidas(ventas)

    FEATURES_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(FEATURES_DATA_FILE)

    print(
        f"[3/3] Guardado {FEATURES_DATA_FILE.relative_to(PROJECT_ROOT)} "
        f"({features.height} clientes, {features.width} columnas)."
    )


if __name__ == "__main__":
    main()

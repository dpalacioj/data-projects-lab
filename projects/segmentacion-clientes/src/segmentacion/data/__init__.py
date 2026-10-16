"""Lectura, carga y limpieza del dataset Online Retail II."""
from segmentacion.data.cargar import ESQUEMA, leer_excel_con_esquema, leer_parquet
from segmentacion.data.limpieza import (
    exigir_cliente,
    filtrar_validas,
    preparar_ventas,
    quitar_duplicados,
    separar_cancelaciones,
)
from segmentacion.data.loader import download_dataset, load_raw_transactions

__all__ = [
    "ESQUEMA",
    "leer_excel_con_esquema",
    "leer_parquet",
    "quitar_duplicados",
    "separar_cancelaciones",
    "exigir_cliente",
    "filtrar_validas",
    "preparar_ventas",
    "download_dataset",
    "load_raw_transactions",
]

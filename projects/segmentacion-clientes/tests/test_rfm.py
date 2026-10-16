"""Tests unitarios de `segmentacion.features.rfm.construir_rfm`.

Rapidos: 5 filas inventadas, una posterior al corte.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from segmentacion.features.rfm import construir_rfm


def _venta(cliente: int, invoice: str, qty: int, price: float, fecha: datetime, stock: str = "85123A") -> dict:
    return {
        "Invoice": invoice,
        "StockCode": stock,
        "Description": "test",
        "Quantity": qty,
        "InvoiceDate": fecha,
        "Price": price,
        "Customer ID": cliente,
        "Country": "United Kingdom",
        "Importe": qty * price,
    }


@pytest.fixture
def ventas_inventadas() -> pl.DataFrame:
    """5 filas: 2 clientes, una fila despues del corte, dos stocks distintos."""
    return pl.DataFrame(
        [
            _venta(1001, "A1", 2, 10.0, datetime(2010, 1, 10), "ITEM-A"),  # dentro
            _venta(1001, "A2", 1, 20.0, datetime(2010, 3, 15), "ITEM-A"),  # dentro
            _venta(1001, "A3", 3, 5.0, datetime(2010, 6, 1), "ITEM-B"),    # dentro
            _venta(1002, "B1", 1, 15.0, datetime(2010, 2, 1), "ITEM-A"),   # dentro
            _venta(1001, "A4", 10, 2.0, datetime(2010, 7, 1), "ITEM-A"),   # POSTERIOR al corte
        ]
    )


def test_construir_rfm_ignora_filas_posteriores_al_corte(ventas_inventadas: pl.DataFrame) -> None:
    corte = datetime(2010, 7, 1)
    rfm = construir_rfm(ventas_inventadas, corte)
    # Cliente 1001: 3 facturas distintas antes del corte; monetario = 20+20+15 = 55.
    fila_1001 = rfm.filter(pl.col("Customer ID") == 1001).row(0, named=True)
    assert fila_1001["frecuencia"] == 3
    assert fila_1001["monetario"] == pytest.approx(55.0)
    assert fila_1001["productos_distintos"] == 2  # ITEM-A y ITEM-B


def test_construir_rfm_devuelve_columnas_esperadas(ventas_inventadas: pl.DataFrame) -> None:
    rfm = construir_rfm(ventas_inventadas, datetime(2011, 1, 1))
    assert set(rfm.columns) == {
        "Customer ID",
        "recencia",
        "antiguedad",
        "frecuencia",
        "monetario",
        "productos_distintos",
    }
    assert rfm.height == 2  # 1001 y 1002


def test_construir_rfm_recencia_y_antiguedad_en_dias(ventas_inventadas: pl.DataFrame) -> None:
    corte = datetime(2010, 7, 1)
    rfm = construir_rfm(ventas_inventadas, corte)
    fila_1002 = rfm.filter(pl.col("Customer ID") == 1002).row(0, named=True)
    # 1002 tiene una sola compra en 2010-02-01; recencia = antiguedad = (2010-07-01) - (2010-02-01) = 150 dias.
    assert fila_1002["recencia"] == 150
    assert fila_1002["antiguedad"] == 150


def test_construir_rfm_corte_igual_a_fecha_excluye_fila(ventas_inventadas: pl.DataFrame) -> None:
    # Corte igual al timestamp de una fila: esa fila debe quedar fuera (filtro `<`).
    corte = datetime(2010, 2, 1)
    rfm = construir_rfm(ventas_inventadas, corte)
    # Cliente 1002 compra justo el 2010-02-01: queda fuera.
    assert rfm.filter(pl.col("Customer ID") == 1002).height == 0

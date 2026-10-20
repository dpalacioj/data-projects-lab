"""Tests de `segmentacion.features.devoluciones.tasa_devolucion`."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from segmentacion.features.devoluciones import tasa_devolucion


@pytest.fixture
def cancelaciones() -> pl.DataFrame:
    """Dos cancelaciones del cliente 1 antes del corte; una posterior al corte."""
    return pl.DataFrame(
        {
            "Invoice": ["C1", "C2", "C3"],
            "InvoiceDate": [
                datetime(2011, 2, 1),
                datetime(2011, 3, 1),
                datetime(2011, 10, 1),
            ],
            "Customer ID": [1, 1, 1],
        }
    )


@pytest.fixture
def frecuencia_por_cliente() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Customer ID": [1, 2, 3],
            "frecuencia": [4, 2, 1],
        }
    )


def test_tasa_divide_por_frecuencia(cancelaciones, frecuencia_por_cliente):
    corte = datetime(2011, 6, 1)
    out = tasa_devolucion(cancelaciones, frecuencia_por_cliente, corte)
    assert out.filter(pl.col("Customer ID") == 1).row(0)[1] == pytest.approx(2 / 4)


def test_cliente_sin_cancelaciones_recibe_cero(cancelaciones, frecuencia_por_cliente):
    corte = datetime(2011, 6, 1)
    out = tasa_devolucion(cancelaciones, frecuencia_por_cliente, corte)
    assert out.filter(pl.col("Customer ID") == 2).row(0)[1] == 0.0
    assert out.filter(pl.col("Customer ID") == 3).row(0)[1] == 0.0


def test_ignora_cancelaciones_posteriores_al_corte(cancelaciones, frecuencia_por_cliente):
    # Corte entre C1 (feb 1) y C2 (mar 1): solo C1 se cuenta pre-corte -> 1/4.
    corte = datetime(2011, 2, 15)
    out = tasa_devolucion(cancelaciones, frecuencia_por_cliente, corte)
    assert out.filter(pl.col("Customer ID") == 1).row(0)[1] == pytest.approx(1 / 4)


def test_columnas_de_salida_son_las_esperadas(cancelaciones, frecuencia_por_cliente):
    corte = datetime(2011, 6, 1)
    out = tasa_devolucion(cancelaciones, frecuencia_por_cliente, corte)
    assert out.columns == ["Customer ID", "tasa_devolucion"]

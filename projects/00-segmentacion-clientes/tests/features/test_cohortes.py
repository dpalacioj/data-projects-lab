"""Tests de `segmentacion.features.cohortes.cohorte`."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from segmentacion.features.cohortes import cohorte


def _venta(cliente, invoice, qty, price, fecha, country="United Kingdom", stock="X") -> dict:
    return {
        "Invoice": invoice,
        "StockCode": stock,
        "Description": "x",
        "Quantity": qty,
        "InvoiceDate": fecha,
        "Price": price,
        "Customer ID": cliente,
        "Country": country,
        "Importe": qty * price,
    }


@pytest.fixture
def datos():
    """Dos clientes con historial variado antes y despues de 2011-06-01."""
    ventas = pl.DataFrame(
        [
            _venta(1, "A1", 2, 10.0, datetime(2011, 1, 1)),  # cliente 1 UK, antes
            _venta(1, "A2", 1, 20.0, datetime(2011, 3, 1)),  # cliente 1 UK, antes
            _venta(1, "A3", 3, 5.0, datetime(2011, 7, 1), stock="Y"),  # 1 recompra
            _venta(2, "B1", 1, 15.0, datetime(2011, 2, 1), country="France"),  # 2 otro, antes
            _venta(2, "B2", 1, 15.0, datetime(2011, 10, 1), country="France"),  # 2 fuera horizonte
        ]
    )
    cancelaciones = pl.DataFrame(
        [
            _venta(1, "CA1", -1, 10.0, datetime(2011, 2, 15)),  # cancelacion valida
        ]
    )
    return ventas, cancelaciones


def test_termina_ordenado_por_customer_id(datos):
    ventas, canc = datos
    c = cohorte(ventas, canc, datetime(2011, 6, 1))
    assert c["Customer ID"].to_list() == sorted(c["Customer ID"].to_list())


def test_pais_mapeado_a_uk_o_otro(datos):
    ventas, canc = datos
    c = cohorte(ventas, canc, datetime(2011, 6, 1))
    paises = set(c["pais"].to_list())
    assert paises <= {"UK", "otro"}
    assert c.filter(pl.col("Customer ID") == 1).row(0, named=True)["pais"] == "UK"
    assert c.filter(pl.col("Customer ID") == 2).row(0, named=True)["pais"] == "otro"


def test_tasa_devolucion_1001_coincide_con_calculo_manual(datos):
    ventas, canc = datos
    c = cohorte(ventas, canc, datetime(2011, 6, 1))
    # Cliente 1 antes del corte: 2 facturas (A1, A2) y 1 cancelacion -> 1/2 = 0.5
    assert c.filter(pl.col("Customer ID") == 1).row(0, named=True)["tasa_devolucion"] == 0.5


def test_recompra_etiquetada_correctamente(datos):
    ventas, canc = datos
    c = cohorte(ventas, canc, datetime(2011, 6, 1), horizonte=90)
    # Cliente 1 compra en 2011-07-01 (dentro) -> recompra=1
    # Cliente 2 solo compra en 2011-10-01 (fuera) -> recompra=0
    assert c.filter(pl.col("Customer ID") == 1).row(0, named=True)["recompra"] == 1
    assert c.filter(pl.col("Customer ID") == 2).row(0, named=True)["recompra"] == 0


def test_recencia_fuga_puede_ser_negativa_si_compra_posterior(datos):
    """recencia_fuga = corte - max(InvoiceDate) sin filtrar: negativa si hay compra futura."""
    ventas, canc = datos
    c = cohorte(ventas, canc, datetime(2011, 6, 1))
    # Cliente 1 tiene compra en 2011-07-01, posterior al corte 2011-06-01 -> negativa.
    assert c.filter(pl.col("Customer ID") == 1).row(0, named=True)["recencia_fuga"] < 0


def test_columnas_de_salida_son_las_esperadas(datos):
    ventas, canc = datos
    c = cohorte(ventas, canc, datetime(2011, 6, 1))
    esperadas = {
        "Customer ID",
        "recencia",
        "antiguedad",
        "frecuencia",
        "monetario",
        "productos_distintos",
        "pais",
        "tasa_devolucion",
        "recompra",
        "recencia_fuga",
    }
    assert set(c.columns) == esperadas

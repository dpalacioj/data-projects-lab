"""Tests de `segmentacion.features.etiquetas.etiqueta_recompra`."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from segmentacion.features.etiquetas import etiqueta_recompra


def _venta(cliente: int, invoice: str, fecha: datetime) -> dict:
    return {
        "Invoice": invoice,
        "StockCode": "X",
        "Description": "x",
        "Quantity": 1,
        "InvoiceDate": fecha,
        "Price": 1.0,
        "Customer ID": cliente,
        "Country": "United Kingdom",
        "Importe": 1.0,
    }


def test_cliente_en_horizonte_queda_marcado():
    corte = datetime(2011, 6, 1)
    ventas = pl.DataFrame(
        [
            _venta(1, "A1", datetime(2011, 7, 1)),  # dentro del horizonte 90d
            _venta(2, "B1", datetime(2011, 3, 1)),  # antes del corte
        ]
    )
    out = etiqueta_recompra(ventas, corte, horizonte_dias=90)
    assert out.height == 1
    assert out.row(0, named=True) == {"Customer ID": 1, "recompra": 1}


def test_cliente_posterior_al_horizonte_no_cuenta():
    corte = datetime(2011, 6, 1)
    ventas = pl.DataFrame(
        [
            _venta(1, "A1", datetime(2011, 10, 1)),  # fuera: 2011-06-01 + 90d = 2011-08-30
        ]
    )
    out = etiqueta_recompra(ventas, corte, horizonte_dias=90)
    assert out.height == 0


def test_horizonte_abierto_a_la_derecha():
    corte = datetime(2011, 6, 1)
    # 2011-06-01 + 90d = 2011-08-30; una venta exactamente ese dia debe quedar fuera.
    ventas = pl.DataFrame(
        [
            _venta(1, "A1", datetime(2011, 8, 30)),
            _venta(2, "B1", datetime(2011, 8, 29)),
        ]
    )
    out = etiqueta_recompra(ventas, corte, horizonte_dias=90)
    clientes_marcados = set(out["Customer ID"].to_list())
    assert clientes_marcados == {2}

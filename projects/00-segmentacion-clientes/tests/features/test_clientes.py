"""Tests unitarios de `segmentacion.features.clientes.construir_features_extendidas`.

Rapidos: 7 filas inventadas, dos clientes, uno fuera del corte.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
from segmentacion.features.clientes import construir_features_extendidas


def _venta(
    cliente: int,
    invoice: str,
    qty: int,
    price: float,
    fecha: datetime,
    stock: str = "85123A",
    pais: str = "United Kingdom",
) -> dict:
    return {
        "Invoice": invoice,
        "StockCode": stock,
        "Description": "test",
        "Quantity": qty,
        "InvoiceDate": fecha,
        "Price": price,
        "Customer ID": cliente,
        "Country": pais,
        "Importe": qty * price,
    }


def _ventas_demo() -> pl.DataFrame:
    """Dos clientes antes del corte, un cliente despues (debe quedar fuera)."""
    return pl.DataFrame(
        [
            _venta(1, "A1", 2, 10.0, datetime(2011, 1, 10), stock="X1"),
            _venta(1, "A2", 4, 5.0, datetime(2011, 2, 5), stock="X2"),
            _venta(1, "A2", 1, 20.0, datetime(2011, 2, 5), stock="X3"),  # misma factura
            _venta(2, "B1", 3, 10.0, datetime(2011, 1, 20), stock="X1", pais="Germany"),
            _venta(2, "B2", 1, 30.0, datetime(2011, 3, 1), stock="X4", pais="Germany"),
            _venta(3, "C1", 5, 10.0, datetime(2011, 7, 1), stock="X1"),  # posterior al corte
        ]
    )


def test_columnas_y_orden_son_los_esperados():
    corte = datetime(2011, 6, 1)
    features = construir_features_extendidas(_ventas_demo(), corte)
    assert features.columns == [
        "Customer ID",
        "Recency",
        "Frequency",
        "Monetary",
        "AvgTicket",
        "ProductDiversity",
        "AvgQuantity",
        "CountryGroup",
    ]


def test_excluye_clientes_sin_actividad_antes_del_corte():
    corte = datetime(2011, 6, 1)
    features = construir_features_extendidas(_ventas_demo(), corte)
    assert features["Customer ID"].to_list() == [1, 2]


def test_valores_derivados_son_correctos_para_cliente_1():
    corte = datetime(2011, 6, 1)
    features = construir_features_extendidas(_ventas_demo(), corte).filter(
        pl.col("Customer ID") == 1
    )
    # Dos facturas (A1, A2), monetario = 20 + 20 + 20 = 60, cantidad total = 7, productos = 3.
    assert features["Frequency"].item() == 2
    assert features["Monetary"].item() == 60.0
    assert features["AvgTicket"].item() == 30.0
    assert features["ProductDiversity"].item() == 3
    assert features["AvgQuantity"].item() == 3.5
    assert features["CountryGroup"].item() == "UK"


def test_country_group_colapsa_no_uk_en_otro():
    corte = datetime(2011, 6, 1)
    features = construir_features_extendidas(_ventas_demo(), corte).filter(
        pl.col("Customer ID") == 2
    )
    assert features["CountryGroup"].item() == "otro"


def test_snapshot_por_defecto_usa_max_fecha_mas_un_dia():
    """Sin fecha_corte explicita, debe incluir al cliente del 2011-07-01."""
    features = construir_features_extendidas(_ventas_demo())
    assert set(features["Customer ID"].to_list()) == {1, 2, 3}

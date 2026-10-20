"""Tests unitarios de `segmentacion.data.limpieza` con datos inventados.

Rapidos (sin tocar disco). Cubren una fila por regla y la secuencia completa.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from segmentacion.data.limpieza import (
    exigir_cliente,
    filtrar_validas,
    preparar_ventas,
    quitar_duplicados,
    separar_cancelaciones,
)


def _fila(invoice: str, cliente: int | None, qty: int, price: float, fecha: datetime) -> dict:
    return {
        "Invoice": invoice,
        "StockCode": "85123A",
        "Description": "test",
        "Quantity": qty,
        "InvoiceDate": fecha,
        "Price": price,
        "Customer ID": cliente,
        "Country": "United Kingdom",
    }


@pytest.fixture
def df_inventado() -> pl.DataFrame:
    """5 filas, una por regla, mas una valida de referencia."""
    base = datetime(2010, 1, 1, 10, 0)
    filas = [
        _fila("536365", 17850, 6, 2.55, base),  # valida
        _fila("536365", 17850, 6, 2.55, base),  # duplicado exacto
        _fila("C536379", 17850, -1, 2.55, base),  # cancelacion
        _fila("536366", None, 6, 2.55, base),  # sin cliente
        _fila("536367", 17850, -3, 2.55, base),  # cantidad invalida
        _fila("536368", 17850, 5, 0.0, base),  # precio invalido
    ]
    return pl.DataFrame(filas)


def test_quitar_duplicados_borra_fila_repetida(df_inventado: pl.DataFrame) -> None:
    sin_dup = quitar_duplicados(df_inventado)
    assert sin_dup.height == df_inventado.height - 1


def test_separar_cancelaciones_devuelve_dos_frames(df_inventado: pl.DataFrame) -> None:
    sin, canc = separar_cancelaciones(df_inventado)
    assert canc.height == 1
    assert canc["Invoice"].item() == "C536379"
    assert sin.height == df_inventado.height - 1
    assert not sin["Invoice"].str.starts_with("C").any()


def test_exigir_cliente_elimina_nulos(df_inventado: pl.DataFrame) -> None:
    con_cliente = exigir_cliente(df_inventado)
    assert con_cliente.height == df_inventado.height - 1
    assert con_cliente["Customer ID"].null_count() == 0


def test_filtrar_validas_descarta_qty_y_precio_no_positivos(df_inventado: pl.DataFrame) -> None:
    validas = filtrar_validas(df_inventado)
    assert validas["Quantity"].min() > 0
    assert validas["Price"].min() > 0
    # Las dos filas invalidas (qty negativa y price cero) y la cancelacion caen.
    assert validas.height == df_inventado.height - 3


def test_preparar_ventas_encadena_las_cuatro_reglas(df_inventado: pl.DataFrame) -> None:
    ventas, cancelaciones = preparar_ventas(df_inventado, log=False)
    # Secuencia esperada: 6 -> 5 (sin duplicado) -> 4 (sin cancelacion) ->
    # 3 (con cliente) -> 1 (cantidad y precio validos).
    assert ventas.height == 1
    assert cancelaciones.height == 1
    assert "Importe" in ventas.columns
    assert ventas["Importe"].item() == pytest.approx(6 * 2.55)
    assert ventas["Invoice"].item() == "536365"


def test_preparar_ventas_funciona_sobre_lazyframe(df_inventado: pl.DataFrame) -> None:
    ventas, cancelaciones = preparar_ventas(df_inventado.lazy(), log=False)
    # Polars 2.x: operaciones lazy devuelven LazyFrame; colectar para verificar.
    assert isinstance(ventas, pl.LazyFrame)
    assert ventas.collect().height == 1
    assert cancelaciones.collect().height == 1

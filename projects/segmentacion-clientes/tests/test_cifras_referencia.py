"""Verifica las cifras de la seccion 4 del SPEC_segmentacion-clientes_polars.

Marcado como `slow`: requiere `data/raw/online_retail_II.parquet`. El fixture
`parquet_real` (en `conftest.py`) hace `skip` si el archivo no existe.

Tolerancias: 0 en conteos exactos, 1e-3 en correlaciones y fracciones.

Si alguna cifra NO coincide:
    1. no modifiques esta prueba;
    2. verifica que no haya cambiado el hash del zip (`descargar_datos.py`);
    3. reporta la discrepancia; actualizar el OVA y la spec se decide a mano.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from segmentacion.data.cargar import leer_parquet
from segmentacion.data.limpieza import (
    exigir_cliente,
    filtrar_validas,
    preparar_ventas,
    quitar_duplicados,
    separar_cancelaciones,
)

pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crudo(parquet_real: Path) -> pl.DataFrame:
    return leer_parquet(parquet_real)


@pytest.fixture(scope="module")
def ventas(crudo: pl.DataFrame) -> pl.DataFrame:
    v, _ = preparar_ventas(crudo, log=False)
    return v


# ----------------------------------------------------------------------------
# Conteos globales
# ----------------------------------------------------------------------------
def test_filas_totales(crudo: pl.DataFrame) -> None:
    assert crudo.height == 1_067_371


def test_filas_sin_customer_id(crudo: pl.DataFrame) -> None:
    assert crudo.filter(pl.col("Customer ID").is_null()).height == 243_007


def test_porcentaje_sin_customer_id(crudo: pl.DataFrame) -> None:
    sin = crudo.filter(pl.col("Customer ID").is_null()).height
    assert abs(sin / crudo.height - 0.2277) < 1e-3


def test_description_nulas(crudo: pl.DataFrame) -> None:
    assert crudo.filter(pl.col("Description").is_null()).height == 4_382


def test_duplicados_exactos(crudo: pl.DataFrame) -> None:
    assert crudo.height - crudo.unique().height == 34_335


def test_facturas_que_empiezan_por_C(crudo: pl.DataFrame) -> None:
    assert crudo.filter(pl.col("Invoice").str.starts_with("C")).height == 19_494


def test_facturas_no_numericas(crudo: pl.DataFrame) -> None:
    # Spec: facturas no numericas (C + A) = 19 500. "No empieza por digito".
    no_num = crudo.filter(~pl.col("Invoice").str.contains(r"^\d")).height
    assert no_num == 19_500


def test_quantity_negativo_sin_factura_C(crudo: pl.DataFrame) -> None:
    cond = (pl.col("Quantity") < 0) & (~pl.col("Invoice").str.starts_with("C"))
    assert crudo.filter(cond).height == 3_457


def test_price_cero(crudo: pl.DataFrame) -> None:
    assert crudo.filter(pl.col("Price") == 0).height == 6_202


def test_price_negativo(crudo: pl.DataFrame) -> None:
    assert crudo.filter(pl.col("Price") < 0).height == 5


def test_stockcode_no_estandar(crudo: pl.DataFrame) -> None:
    # Spec: no `^\d{5}` (sin anclaje final: no empieza con 5 digitos).
    assert crudo.filter(~pl.col("StockCode").str.contains(r"^\d{5}")).height == 6_093


def test_paises_unicos(crudo: pl.DataFrame) -> None:
    assert crudo.select(pl.col("Country").n_unique()).item() == 43


def test_clientes_unicos(crudo: pl.DataFrame) -> None:
    # n_unique incluye el valor null como una categoria; spec reporta 5 943.
    assert crudo.select(pl.col("Customer ID").n_unique()).item() == 5_943


def test_productos_unicos(crudo: pl.DataFrame) -> None:
    assert crudo.select(pl.col("StockCode").n_unique()).item() == 5_305


def test_facturas_unicas(crudo: pl.DataFrame) -> None:
    assert crudo.select(pl.col("Invoice").n_unique()).item() == 53_628


# ----------------------------------------------------------------------------
# Pipeline secuencial (seccion 4, fila "Pipeline secuencial")
# ----------------------------------------------------------------------------
def test_pipeline_secuencial_tras_unique(crudo: pl.DataFrame) -> None:
    assert quitar_duplicados(crudo).height == 1_033_036


def test_pipeline_secuencial_tras_quitar_C(crudo: pl.DataFrame) -> None:
    sin_dup = quitar_duplicados(crudo)
    sin_c, _ = separar_cancelaciones(sin_dup)
    assert sin_c.height == 1_013_932


def test_pipeline_secuencial_tras_exigir_cliente(crudo: pl.DataFrame) -> None:
    sin_dup = quitar_duplicados(crudo)
    sin_c, _ = separar_cancelaciones(sin_dup)
    assert exigir_cliente(sin_c).height == 779_495


def test_pipeline_secuencial_final(ventas: pl.DataFrame) -> None:
    assert ventas.height == 779_425
    assert abs(ventas.height / 1_067_371 - 0.730) < 1e-3


# ----------------------------------------------------------------------------
# Estadisticas sobre ventas validas
# ----------------------------------------------------------------------------
def test_price_mediana(ventas: pl.DataFrame) -> None:
    assert ventas["Price"].median() == pytest.approx(1.95, abs=1e-3)


def test_price_q3(ventas: pl.DataFrame) -> None:
    assert ventas["Price"].quantile(0.75) == pytest.approx(3.75, abs=1e-3)


def test_price_media(ventas: pl.DataFrame) -> None:
    # El OVA reporta la media con dos decimales (3,22). Tolerancia = media-LSD.
    assert ventas["Price"].mean() == pytest.approx(3.22, abs=5e-3)


def test_price_max(ventas: pl.DataFrame) -> None:
    assert ventas["Price"].max() == pytest.approx(10_953.50, abs=1e-3)


def test_quantity_mediana(ventas: pl.DataFrame) -> None:
    assert ventas["Quantity"].median() == 6


def test_quantity_max(ventas: pl.DataFrame) -> None:
    assert ventas["Quantity"].max() == 80_995


# ----------------------------------------------------------------------------
# Cercas de Tukey y correlaciones
# ----------------------------------------------------------------------------
def test_tukey_price(ventas: pl.DataFrame) -> None:
    q1 = ventas["Price"].quantile(0.25)
    q3 = ventas["Price"].quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    assert fence == pytest.approx(7.50, abs=1e-3)
    frac = (ventas["Price"] > fence).sum() / ventas.height
    assert abs(frac - 0.084) < 1e-3


def test_tukey_quantity(ventas: pl.DataFrame) -> None:
    q1 = ventas["Quantity"].quantile(0.25)
    q3 = ventas["Quantity"].quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    assert fence == pytest.approx(27.0, abs=1e-3)
    frac = (ventas["Quantity"] > fence).sum() / ventas.height
    assert abs(frac - 0.0656) < 1e-3


def test_correlaciones_quantity_price(ventas: pl.DataFrame) -> None:
    pearson = ventas.select(pl.corr("Quantity", "Price")).item()
    spearman = ventas.select(pl.corr("Quantity", "Price", method="spearman")).item()
    assert abs(pearson - (-0.005)) < 1e-3
    assert abs(spearman - (-0.428)) < 1e-3


def test_spearman_frecuencia_monetario(ventas: pl.DataFrame) -> None:
    rfm = ventas.group_by("Customer ID").agg(
        pl.col("Invoice").n_unique().alias("frecuencia"),
        pl.col("Importe").sum().alias("monetario"),
    )
    rho = rfm.select(pl.corr("frecuencia", "monetario", method="spearman")).item()
    assert abs(rho - 0.858) < 1e-3


def test_participacion_top_20_porciento(ventas: pl.DataFrame) -> None:
    rfm = (
        ventas.group_by("Customer ID")
        .agg(pl.col("Importe").sum().alias("monetario"))
        .sort("monetario", descending=True)
    )
    n_top = int(0.2 * rfm.height)
    share = rfm.head(n_top)["monetario"].sum() / rfm["monetario"].sum()
    assert abs(share - 0.772) < 1e-3


# ----------------------------------------------------------------------------
# Temporalidad
# ----------------------------------------------------------------------------
def test_filas_sabado(ventas: pl.DataFrame) -> None:
    # Polars weekday: lun=1..dom=7. Sabado = 6.
    sab = ventas.filter(pl.col("InvoiceDate").dt.weekday() == 6).height
    assert sab == 400


def test_rango_fechas(ventas: pl.DataFrame) -> None:
    assert ventas["InvoiceDate"].min() == datetime(2009, 12, 1, 7, 45)
    assert ventas["InvoiceDate"].max() == datetime(2011, 12, 9, 12, 50)

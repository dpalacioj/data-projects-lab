"""Tests del modulo LLM: sin red, sin claves (`FakeLLM`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from segmentacion.features.llm_clientes import FakeLLM
from segmentacion.features.texto_llm import (
    CATEGORIAS_LISTA,
    categorizar_descripciones,
    evaluar,
    proporcion_gasto_por_categoria,
    regla_palabras_clave,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "respuestas_fake.json"


@pytest.fixture
def cliente_fake() -> FakeLLM:
    return FakeLLM(FIXTURE)


def _descripciones_lote() -> pl.Series:
    return pl.Series(
        "Description",
        [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
        ],
    )


def test_categorizar_usa_cache(tmp_path, cliente_fake):
    """La segunda llamada con la misma cache no debe invocar al cliente."""
    cache = tmp_path / "categorias.parquet"

    # Primera llamada: el cliente responde para cada descripcion.
    out1 = categorizar_descripciones(
        _descripciones_lote(), cache=cache, cliente=cliente_fake, lote=1
    )
    assert out1.height == 3
    assert cache.exists()

    # Segunda llamada: envolvemos cliente_fake con un MagicMock que delega;
    # verificamos que no se invoca.
    espia = MagicMock(side_effect=cliente_fake)
    out2 = categorizar_descripciones(_descripciones_lote(), cache=cache, cliente=espia, lote=1)
    assert out2.height == 3
    espia.assert_not_called()


def test_salida_invalida_cae_en_otro(tmp_path, cliente_fake):
    """Una etiqueta inexistente debe caer en 'otro' con fuente='invalido'."""
    cache = tmp_path / "categorias.parquet"
    out = categorizar_descripciones(
        pl.Series("Description", ["ETIQUETA_INEXISTENTE"]),
        cache=cache,
        cliente=cliente_fake,
        lote=1,
        reintentos=0,
    )
    fila = out.row(0, named=True)
    assert fila["categoria"] == "otro"
    assert fila["fuente"] == "invalido"


def test_json_malformado_reintenta_y_cae_en_invalido(tmp_path, cliente_fake):
    """Respuesta no parseable → tras reintentos → `invalido`."""
    cache = tmp_path / "categorias.parquet"
    out = categorizar_descripciones(
        pl.Series("Description", ["MALFORMED"]),
        cache=cache,
        cliente=cliente_fake,
        lote=1,
        reintentos=2,
    )
    assert out.row(0, named=True)["fuente"] == "invalido"


def test_proporciones_suman_uno(tmp_path):
    """Por cliente, las 12 columnas prop_* suman 1 ± 1e-9."""
    categorias = pl.DataFrame(
        {
            "Description": ["A", "B", "C"],
            "categoria": ["cocina", "textil", "cocina"],
        }
    )
    ventas = pl.DataFrame(
        {
            "Customer ID": [1, 1, 2],
            "InvoiceDate": [datetime(2011, 1, 1), datetime(2011, 2, 1), datetime(2011, 3, 1)],
            "Description": ["A", "B", "C"],
            "Importe": [10.0, 30.0, 5.0],
        }
    )
    out = proporcion_gasto_por_categoria(ventas, categorias, datetime(2011, 6, 1))
    cols_prop = [f"prop_{c}" for c in CATEGORIAS_LISTA]
    for fila in out.iter_rows(named=True):
        total = sum(fila[c] for c in cols_prop)
        assert abs(total - 1.0) < 1e-9


def test_sin_fuga_firma_no_acepta_cliente_fecha_ni_etiqueta():
    """La firma de `categorizar_descripciones` no incluye datos de cliente."""
    import inspect

    sig = inspect.signature(categorizar_descripciones)
    assert "Customer ID" not in sig.parameters
    assert "fecha_corte" not in sig.parameters
    assert "recompra" not in sig.parameters
    assert "etiqueta" not in sig.parameters


def test_regla_palabras_clave_detecta_categorias():
    """Linea base: palabras clave sin LLM."""
    out = regla_palabras_clave(
        pl.Series(
            "Description",
            [
                "WHITE CANDLE HOLDER",
                "RED COFFEE MUG",
                "CHRISTMAS WREATH",
                "JUMBO BAG",
                "POSTAGE",
                "ALGO EXTRANO",
            ],
        )
    )
    categorias = out["categoria"].to_list()
    assert categorias[0] == "iluminacion"
    assert categorias[1] == "cocina"
    assert categorias[2] == "navidad"
    assert categorias[3] == "bolsas"
    assert categorias[4] == "cargo_no_producto"
    assert categorias[5] == "otro"


def test_evaluar_devuelve_exactitud_contra_gold():
    categorias = pl.DataFrame(
        {
            "Description": ["A", "B", "C"],
            "categoria": ["cocina", "textil", "otro"],
        }
    )
    gold = pl.DataFrame(
        {
            "Description": ["A", "B", "C"],
            "categoria": ["cocina", "textil", "cocina"],
        }
    )
    r = evaluar(categorias, gold)
    assert r["n"] == 3
    assert r["exactitud"] == pytest.approx(2 / 3)

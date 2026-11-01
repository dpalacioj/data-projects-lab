"""`Predictor`: lee todo de los metadatos y responde igual por cualquier camino."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from segmentacion.servicio import EntradaCliente, Predictor

from .conftest import CASO

# Cifras de referencia del OVA U4 (Tabla 2), umbral 0,3. Tolerancia 5e-4 (4 decimales).
CASOS_REFERENCIA = [
    ((12000, 30, 150, 5, 540, 0.10, "Germany"), 0.9622),
    ((2350, 9, 60, 12, 400, 0.0, "United Kingdom"), 0.7477),
    ((900, 3, 25, 60, 300, 0.33, "United Kingdom"), 0.3577),
    ((900, 3, 25, 60, 300, 0.0, "United Kingdom"), 0.3545),
    ((400, 2, 15, 120, 380, 0.5, "UK"), 0.1357),
    ((180, 1, 8, 210, 210, 0.0, "France"), 0.0463),
]


def _como_dict(valores: tuple) -> dict:
    return dict(zip(list(EntradaCliente.model_fields), valores, strict=True))


def test_predictor_lee_variables_y_umbral_de_los_metadatos(artefacto_chico: Path):
    p = Predictor.desde_carpeta(artefacto_chico)
    assert p.variables == p.metadatos["variables_en_orden"]
    assert p.umbral_acordado == p.metadatos["umbral_acordado"]
    assert p.version_modelo == "recompra_test"


def test_predecir_devuelve_contrato_y_aplica_umbral(artefacto_chico: Path):
    p = Predictor.desde_carpeta(artefacto_chico)
    r = p.predecir(CASO)
    assert set(r) == {"probabilidad", "recompra", "umbral", "version_modelo"}
    assert 0.0 <= r["probabilidad"] <= 1.0
    assert r["umbral"] == 0.3
    # con umbral 1,0 nadie recompra; con 0,0 todos
    assert p.predecir(CASO, umbral=1.0)["recompra"] is False
    assert p.predecir(CASO, umbral=0.0)["recompra"] is True


def test_dict_y_entrada_cliente_dan_lo_mismo(artefacto_chico: Path):
    p = Predictor.desde_carpeta(artefacto_chico)
    assert p.predecir(CASO) == p.predecir(EntradaCliente(**CASO))


def test_predecir_lote_coincide_con_predecir(artefacto_chico: Path):
    p = Predictor.desde_carpeta(artefacto_chico)
    df = pl.DataFrame([_como_dict(c) for c, _ in CASOS_REFERENCIA])
    lote = p.predecir_lote(df)
    assert lote.height == df.height
    assert lote["probabilidad"].is_sorted(descending=True)
    for fila in lote.iter_rows(named=True):
        uno = p.predecir({k: fila[k] for k in p.variables})
        assert abs(fila["probabilidad"] - uno["probabilidad"]) < 1e-6
        assert fila["recompra"] == uno["recompra"]


def test_salud_informa_version_y_sklearn(artefacto_chico: Path):
    s = Predictor.desde_carpeta(artefacto_chico).salud()
    assert s["estado"] == "ok"
    assert s["version_modelo"] == "recompra_test"
    assert s["fecha_corte"] == "2011-06-01"
    assert "sklearn" in s


@pytest.mark.slow
@pytest.mark.parametrize("valores, esperado", CASOS_REFERENCIA)
def test_cifras_de_referencia_del_ova(artefacto_real: Path, valores: tuple, esperado: float):
    """Tabla 2 del OVA U4 contra el artefacto real entrenado con `scripts/entrenar_recompra.py`."""
    r = Predictor.desde_carpeta(artefacto_real).predecir(_como_dict(valores))
    assert abs(r["probabilidad"] - esperado) < 5e-4

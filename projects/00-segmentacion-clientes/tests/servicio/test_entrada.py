"""`EntradaCliente` es la unica copia de la lista de variables fuera de los metadatos."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from segmentacion.pipelines.preprocesamiento import CAT, NUM_LOG, NUM_STD
from segmentacion.servicio import EntradaCliente

from .conftest import CASO


def test_entrada_sigue_el_orden_de_los_metadatos(metadatos_versionados: dict):
    """Mismo nombre y mismo orden que `metadatos.json["variables_en_orden"]` (versionado)."""
    assert list(EntradaCliente.model_fields) == metadatos_versionados["variables_en_orden"]


def test_entrada_sigue_el_orden_del_preprocesador():
    assert list(EntradaCliente.model_fields) == NUM_LOG + NUM_STD + CAT


@pytest.mark.parametrize("pais", ["UK", "United Kingdom", "reino unido", " united kingdom "])
def test_pais_se_normaliza_a_uk(pais: str):
    assert EntradaCliente(**{**CASO, "pais": pais}).pais == "UK"


@pytest.mark.parametrize("pais", ["France", "Germany", "Otro país"])
def test_pais_distinto_se_normaliza_a_otro(pais: str):
    assert EntradaCliente(**{**CASO, "pais": pais}).pais == "otro"


@pytest.mark.parametrize(
    "campo, valor",
    [("recencia", -3), ("monetario", -1.0), ("frecuencia", 0), ("productos_distintos", 0)],
)
def test_entrada_invalida_se_rechaza(campo: str, valor):
    with pytest.raises(ValidationError):
        EntradaCliente(**{**CASO, campo: valor})

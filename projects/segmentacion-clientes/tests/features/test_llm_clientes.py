"""Tests de `segmentacion.features.llm_clientes` sin red."""

from __future__ import annotations

from pathlib import Path

import pytest
from segmentacion.features.llm_clientes import FakeLLM, cliente_openai_compatible

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "respuestas_fake.json"


def test_fakellm_responde_segun_substring():
    cliente = FakeLLM(FIXTURE)
    respuesta = cliente("categoriza WHITE HANGING HEART T-LIGHT HOLDER")
    assert "iluminacion" in respuesta


def test_fakellm_devuelve_vacio_si_no_hay_match():
    cliente = FakeLLM(FIXTURE)
    respuesta = cliente("texto que no aparece en el fixture")
    assert respuesta == "[]"


def test_cliente_openai_sin_api_key_falla_claro(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        cliente_openai_compatible(
            base_url="https://api.ejemplo.com",
            api_key_env="LLM_API_KEY",
            modelo="algun-modelo",
        )

"""Tests de la factory de LLMs."""
import pytest
from chat_bd.llm import obtener_llm


def test_factory_fake_responde_lo_que_le_damos():
    llm = obtener_llm(proveedor="fake", respuestas_fake=["hola mundo"])
    assert llm.invoke("cualquier cosa").content == "hola mundo"


def test_factory_fake_sin_respuestas_falla():
    with pytest.raises(ValueError):
        obtener_llm(proveedor="fake")


def test_factory_proveedor_desconocido_falla():
    with pytest.raises(ValueError):
        obtener_llm(proveedor="chatgpt-plus")


def test_factory_hf_sin_token_falla(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from chat_bd import config
    config.HF_TOKEN = None

    with pytest.raises(ValueError, match="HF_TOKEN"):
        obtener_llm(proveedor="hf")

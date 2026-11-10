"""Tests del esquema Pydantic: validaciones y errores claros."""
import pytest
from ingesta_eventos.schemas import LoteJSON, PedidoJSON
from pydantic import ValidationError


def test_pedido_valido_se_parsea(pedido_json_valido):
    pedido = PedidoJSON.model_validate(pedido_json_valido)
    assert pedido.pedido_id == "P-TEST-00001"
    assert pedido.canal == "web"
    assert len(pedido.items) == 2


def test_lote_valido_se_parsea(lote_json_valido):
    lote = LoteJSON.model_validate(lote_json_valido)
    assert len(lote.pedidos) == 2


def test_pedido_rechaza_canal_invalido(pedido_json_valido):
    pedido_json_valido["canal"] = "telefono"  # no esta en el Literal
    with pytest.raises(ValidationError):
        PedidoJSON.model_validate(pedido_json_valido)


def test_pedido_rechaza_cantidad_negativa(pedido_json_valido):
    pedido_json_valido["items"][0]["cantidad"] = -1
    with pytest.raises(ValidationError):
        PedidoJSON.model_validate(pedido_json_valido)


def test_pedido_rechaza_email_sin_arroba(pedido_json_valido):
    pedido_json_valido["cliente"]["email"] = "sin-arroba"
    with pytest.raises(ValidationError):
        PedidoJSON.model_validate(pedido_json_valido)


def test_pedido_requiere_al_menos_un_item(pedido_json_valido):
    pedido_json_valido["items"] = []
    with pytest.raises(ValidationError):
        PedidoJSON.model_validate(pedido_json_valido)

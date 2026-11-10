"""Tests del aplanado: JSON anidado -> 4 listas normalizadas."""
from ingesta_eventos.schemas import LoteJSON
from ingesta_eventos.transform import aplanar_lote


def test_aplanar_devuelve_cuatro_claves(lote_json_valido):
    lote = LoteJSON.model_validate(lote_json_valido)
    out = aplanar_lote(lote)
    assert set(out.keys()) == {"clientes", "productos", "pedidos", "items"}


def test_aplanar_deduplica_clientes_y_productos(lote_json_valido):
    """Dos pedidos del mismo cliente con los mismos productos = 1 cliente, 2 productos."""
    lote = LoteJSON.model_validate(lote_json_valido)
    out = aplanar_lote(lote)
    assert len(out["clientes"]) == 1
    assert len(out["productos"]) == 2
    assert len(out["pedidos"]) == 2
    assert len(out["items"]) == 4  # 2 pedidos x 2 items


def test_pedido_lleva_total_y_n_items_precalculados(lote_json_valido):
    lote = LoteJSON.model_validate(lote_json_valido)
    out = aplanar_lote(lote)
    pedido = out["pedidos"][0]
    # 2 auriculares a 50.0 + 1 camiseta a 25.5 = 125.50
    assert pedido["total"] == 125.50
    assert pedido["n_items"] == 2


def test_items_llevan_subtotal(lote_json_valido):
    lote = LoteJSON.model_validate(lote_json_valido)
    out = aplanar_lote(lote)
    # El primer item: cantidad=2, precio=50 -> subtotal=100
    assert out["items"][0]["subtotal"] == 100.0


def test_totales_coinciden_con_suma_de_items(lote_json_valido):
    lote = LoteJSON.model_validate(lote_json_valido)
    out = aplanar_lote(lote)
    for pedido in out["pedidos"]:
        items_del_pedido = [i for i in out["items"] if i["pedido_id"] == pedido["pedido_id"]]
        suma = round(sum(i["subtotal"] for i in items_del_pedido), 2)
        assert suma == pedido["total"]

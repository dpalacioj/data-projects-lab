"""Fixtures compartidas por los tests.

El proyecto normalmente apunta a un SQLite en `data/db/eventos.db`. Para los
tests levantamos un SQLite en memoria (`sqlite:///:memory:`) con tablas
limpias en cada caso. Asi los tests son:

- rapidos (sin I/O a disco)
- aislados (no dependen de `make generate-data` ni `make load`)
- seguros (nunca tocan la base del usuario)
"""
from __future__ import annotations

import pytest
from ingesta_eventos.db import crear_engine, crear_tablas


@pytest.fixture
def engine_memoria():
    """Engine SQLite en memoria con tablas ya creadas."""
    engine = crear_engine("sqlite:///:memory:")
    crear_tablas(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def pedido_json_valido() -> dict:
    """Un pedido de ejemplo que respeta el esquema."""
    return {
        "pedido_id": "P-TEST-00001",
        "fecha": "2024-01-15T10:32:00",
        "canal": "web",
        "cliente": {
            "cliente_id": "C-000001",
            "nombre": "Ana Garcia",
            "email": "ana.garcia@mail.test",
            "ciudad": "Medellin",
            "pais": "Colombia",
        },
        "items": [
            {
                "producto_id": "SKU-00001",
                "nombre": "Auriculares Electronica #0001",
                "categoria": "Electronica",
                "cantidad": 2,
                "precio_unitario": 50.0,
            },
            {
                "producto_id": "SKU-00002",
                "nombre": "Camiseta Moda #0002",
                "categoria": "Moda",
                "cantidad": 1,
                "precio_unitario": 25.5,
            },
        ],
    }


@pytest.fixture
def lote_json_valido(pedido_json_valido) -> dict:
    """Un lote minimo con dos pedidos, reutilizando el cliente y los productos."""
    pedido_2 = {
        **pedido_json_valido,
        "pedido_id": "P-TEST-00002",
        "fecha": "2024-01-15T11:00:00",
        "canal": "app",
    }
    return {
        "fecha_lote": "2024-01-15",
        "version_esquema": "1.0",
        "pedidos": [pedido_json_valido, pedido_2],
    }

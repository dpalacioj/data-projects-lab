"""Aplanado del JSON anidado a filas listas para insertar.

Un `PedidoJSON` tiene cliente anidado y una lista de items anidados. En la
base lo partimos en 4 tablas normalizadas:

    clientes     (cliente_id, nombre, email, ciudad, pais)
    productos    (producto_id, nombre, categoria)
    pedidos      (pedido_id, cliente_id, fecha, canal, total, n_items)
    items_pedido (item_id, pedido_id, producto_id, cantidad, precio, subtotal)

La funcion `aplanar_lote` recibe un `LoteJSON` y devuelve 4 listas de dicts.
No toca la base. Esto es importante: la transformacion es pura (misma entrada
=> misma salida), lo que hace que se pueda testear sin montar BD alguna.
"""
from __future__ import annotations

from typing import TypedDict

from .schemas import LoteJSON


class FilasAplanadas(TypedDict):
    clientes: list[dict]
    productos: list[dict]
    pedidos: list[dict]
    items: list[dict]


def aplanar_lote(lote: LoteJSON) -> FilasAplanadas:
    """Convierte un lote JSON a 4 listas de dicts, deduplicando clientes y productos.

    - `clientes` y `productos` quedan deduplicados DENTRO del lote (puede
      haber duplicados ENTRE lotes; eso se maneja al cargar).
    - `pedidos` lleva `total` y `n_items` precalculados (patron "denormalizacion
      controlada": se guarda el resumen para evitar recalcular en cada query).
    - `items` lleva `subtotal = cantidad * precio_unitario` precalculado.
    """
    clientes: dict[str, dict] = {}
    productos: dict[str, dict] = {}
    pedidos: list[dict] = []
    items: list[dict] = []

    for pedido in lote.pedidos:
        # Cliente: un pedido = un cliente. Dedup por cliente_id.
        c = pedido.cliente
        clientes.setdefault(
            c.cliente_id,
            {
                "cliente_id": c.cliente_id,
                "nombre": c.nombre,
                "email": c.email,
                "ciudad": c.ciudad,
                "pais": c.pais,
            },
        )

        total = 0.0
        for item in pedido.items:
            productos.setdefault(
                item.producto_id,
                {
                    "producto_id": item.producto_id,
                    "nombre": item.nombre,
                    "categoria": item.categoria,
                },
            )
            subtotal = round(item.cantidad * item.precio_unitario, 2)
            total += subtotal
            items.append(
                {
                    "pedido_id": pedido.pedido_id,
                    "producto_id": item.producto_id,
                    "cantidad": item.cantidad,
                    "precio_unitario": item.precio_unitario,
                    "subtotal": subtotal,
                }
            )

        pedidos.append(
            {
                "pedido_id": pedido.pedido_id,
                "cliente_id": c.cliente_id,
                "fecha": pedido.fecha,
                "canal": pedido.canal,
                "total": round(total, 2),
                "n_items": len(pedido.items),
            }
        )

    return {
        "clientes": list(clientes.values()),
        "productos": list(productos.values()),
        "pedidos": pedidos,
        "items": items,
    }

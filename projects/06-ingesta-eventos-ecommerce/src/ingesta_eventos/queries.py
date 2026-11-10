"""Consultas de negocio sobre la base ya cargada.

El dashboard (`app/streamlit_app.py`) solo llama funciones de aqui. La logica
SQL vive en UN SOLO SITIO; si una query cambia, se arregla una sola vez.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def conteos_tablas(engine: Engine) -> dict[str, int]:
    """Numero de filas por tabla. Primer sanity check tras una carga."""
    sql = {
        "clientes": "SELECT COUNT(*) FROM clientes",
        "productos": "SELECT COUNT(*) FROM productos",
        "pedidos": "SELECT COUNT(*) FROM pedidos",
        "items_pedido": "SELECT COUNT(*) FROM items_pedido",
    }
    out = {}
    with engine.connect() as conn:
        for nombre, q in sql.items():
            out[nombre] = int(conn.execute(text(q)).scalar() or 0)
    return out


def ingresos_por_dia(engine: Engine) -> pd.DataFrame:
    """Serie temporal de ingresos diarios."""
    sql = text(
        """
        SELECT DATE(fecha) AS dia,
               SUM(total)  AS ingresos,
               COUNT(*)    AS pedidos
        FROM pedidos
        GROUP BY DATE(fecha)
        ORDER BY dia
        """
    )
    return pd.read_sql(sql, engine)


def top_productos(engine: Engine, limite: int = 10) -> pd.DataFrame:
    """Top productos por ingresos acumulados."""
    sql = text(
        """
        SELECT p.producto_id,
               p.nombre,
               p.categoria,
               SUM(i.cantidad)  AS unidades_vendidas,
               SUM(i.subtotal)  AS ingresos
        FROM items_pedido i
        JOIN productos p ON p.producto_id = i.producto_id
        GROUP BY p.producto_id, p.nombre, p.categoria
        ORDER BY ingresos DESC
        LIMIT :limite
        """
    )
    return pd.read_sql(sql, engine, params={"limite": limite})


def ingresos_por_canal(engine: Engine) -> pd.DataFrame:
    """Reparto de ingresos y pedidos por canal (web / app / marketplace)."""
    sql = text(
        """
        SELECT canal,
               COUNT(*)   AS pedidos,
               SUM(total) AS ingresos,
               AVG(total) AS ticket_promedio
        FROM pedidos
        GROUP BY canal
        ORDER BY ingresos DESC
        """
    )
    return pd.read_sql(sql, engine)


def clientes_recurrentes(engine: Engine, limite: int = 10) -> pd.DataFrame:
    """Clientes con mas pedidos (segmento 'fieles' en una RFM simple)."""
    sql = text(
        """
        SELECT c.cliente_id,
               c.nombre,
               c.ciudad,
               c.pais,
               COUNT(*)   AS pedidos,
               SUM(p.total) AS gasto_total
        FROM pedidos p
        JOIN clientes c ON c.cliente_id = p.cliente_id
        GROUP BY c.cliente_id, c.nombre, c.ciudad, c.pais
        ORDER BY pedidos DESC, gasto_total DESC
        LIMIT :limite
        """
    )
    return pd.read_sql(sql, engine, params={"limite": limite})


def ingresos_por_pais(engine: Engine) -> pd.DataFrame:
    sql = text(
        """
        SELECT c.pais,
               COUNT(DISTINCT c.cliente_id) AS clientes,
               COUNT(*)                     AS pedidos,
               SUM(p.total)                 AS ingresos
        FROM pedidos p
        JOIN clientes c ON c.cliente_id = p.cliente_id
        GROUP BY c.pais
        ORDER BY ingresos DESC
        """
    )
    return pd.read_sql(sql, engine)

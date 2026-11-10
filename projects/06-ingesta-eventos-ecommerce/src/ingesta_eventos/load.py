"""Carga de las filas aplanadas a la base.

Dos ideas clave:

1. Carga en batches. Un `insert(table).values([...])` con 5.000 dicts es
   ordenes de magnitud mas rapido que 5.000 llamadas separadas.
2. Idempotencia. `clientes` y `productos` pueden repetirse entre lotes; usamos
   `INSERT OR IGNORE` (SQLite) / `ON CONFLICT DO NOTHING` (Postgres) para que
   rerunning el pipeline no duplique.
"""
from __future__ import annotations

from typing import Iterable

from sqlalchemy import Table, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from .config import BATCH_SIZE
from .db import clientes_table, items_table, pedidos_table, productos_table
from .transform import FilasAplanadas


def _upsert_ignorando_duplicados(engine: Engine, tabla: Table, filas: list[dict]) -> int:
    """INSERT ... ON CONFLICT DO NOTHING para SQLite y Postgres.

    Devuelve el numero de filas enviadas (no el numero efectivamente insertado:
    para eso habria que leer `rowcount` que es poco fiable en SQLite).
    """
    if not filas:
        return 0

    dialecto = engine.dialect.name
    enviadas = 0

    with engine.begin() as conn:
        for i in range(0, len(filas), BATCH_SIZE):
            trozo = filas[i : i + BATCH_SIZE]
            if dialecto == "sqlite":
                stmt = sqlite_insert(tabla).values(trozo).prefix_with("OR IGNORE")
            elif dialecto == "postgresql":
                stmt = pg_insert(tabla).values(trozo).on_conflict_do_nothing()
            else:
                stmt = insert(tabla).values(trozo)
            conn.execute(stmt)
            enviadas += len(trozo)

    return enviadas


def _insert_simple(engine: Engine, tabla: Table, filas: list[dict]) -> int:
    """INSERT puro, por batches. Para tablas sin PK natural que puedan colisionar."""
    if not filas:
        return 0
    enviadas = 0
    with engine.begin() as conn:
        for i in range(0, len(filas), BATCH_SIZE):
            trozo = filas[i : i + BATCH_SIZE]
            conn.execute(insert(tabla), trozo)
            enviadas += len(trozo)
    return enviadas


def cargar_lote(engine: Engine, filas: FilasAplanadas) -> dict:
    """Carga a la base las 4 tablas de un lote aplanado. Devuelve conteos."""
    return {
        "clientes": _upsert_ignorando_duplicados(engine, clientes_table, filas["clientes"]),
        "productos": _upsert_ignorando_duplicados(engine, productos_table, filas["productos"]),
        "pedidos": _upsert_ignorando_duplicados(engine, pedidos_table, filas["pedidos"]),
        "items": _insert_simple(engine, items_table, filas["items"]),
    }


def resumir(resultados: Iterable[dict]) -> dict:
    """Suma conteos de varios lotes. Util para reportar al final del pipeline."""
    total = {"clientes": 0, "productos": 0, "pedidos": 0, "items": 0}
    for r in resultados:
        for k in total:
            total[k] += r.get(k, 0)
    return total

"""Conexion a la base y DDL.

Toda la capa de BD pasa por aqui: nadie mas en el paquete hace `create_engine`,
nadie mas define tablas. Un solo lugar para cambiar de SQLite a Postgres.

Por que SQLAlchemy Core y no un ORM:
    Para ETL masivo (1M+ filas) el ORM es lento: por cada fila se instancian
    objetos Python. Core usa `insert()` crudo y acepta batches de miles de
    filas por llamada. Es el mismo patron que usa un pipeline real.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Engine

from .config import (
    DATABASE_URL,
    TABLA_CLIENTES,
    TABLA_ITEMS,
    TABLA_PEDIDOS,
    TABLA_PRODUCTOS,
    database_url_safe,
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Definicion de tablas (DDL declarativo con SQLAlchemy Core)
# ---------------------------------------------------------------------------
clientes_table = Table(
    TABLA_CLIENTES,
    metadata,
    Column("cliente_id", String(32), primary_key=True),
    Column("nombre", String(128), nullable=False),
    Column("email", String(128), nullable=False),
    Column("ciudad", String(64)),
    Column("pais", String(64)),
)

productos_table = Table(
    TABLA_PRODUCTOS,
    metadata,
    Column("producto_id", String(32), primary_key=True),
    Column("nombre", String(128), nullable=False),
    Column("categoria", String(64)),
)

pedidos_table = Table(
    TABLA_PEDIDOS,
    metadata,
    Column("pedido_id", String(32), primary_key=True),
    Column("cliente_id", String(32), ForeignKey(f"{TABLA_CLIENTES}.cliente_id"), nullable=False),
    Column("fecha", DateTime, nullable=False, index=True),
    Column("canal", String(16), nullable=False, index=True),
    Column("total", Float, nullable=False),
    Column("n_items", Integer, nullable=False),
)

items_table = Table(
    TABLA_ITEMS,
    metadata,
    Column("item_id", Integer, primary_key=True, autoincrement=True),
    Column("pedido_id", String(32), ForeignKey(f"{TABLA_PEDIDOS}.pedido_id"), nullable=False),
    Column("producto_id", String(32), ForeignKey(f"{TABLA_PRODUCTOS}.producto_id"), nullable=False),
    Column("cantidad", Integer, nullable=False),
    Column("precio_unitario", Float, nullable=False),
    Column("subtotal", Float, nullable=False),
)


# ---------------------------------------------------------------------------
# Engine + optimizaciones
# ---------------------------------------------------------------------------
def crear_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Crea el engine de SQLAlchemy a partir de la URL (o la del entorno).

    Para SQLite activa WAL + synchronous=NORMAL: subida de rendimiento de
    cargas masivas sin sacrificar durabilidad razonable.
    """
    url = url or DATABASE_URL
    # `future=True` es el default en 2.x pero lo dejamos explicito.
    engine = create_engine(url, echo=echo, future=True)

    if url.startswith("sqlite"):
        _configurar_sqlite(engine)

    return engine


def _configurar_sqlite(engine: Engine) -> None:
    """PRAGMAs recomendadas para cargas batch en SQLite."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.close()


def crear_tablas(engine: Engine) -> None:
    """Ejecuta el DDL: crea las 4 tablas si no existen."""
    metadata.create_all(engine)


def borrar_tablas(engine: Engine) -> None:
    """Dropea todo. Util en tests y en `make clean-db`."""
    metadata.drop_all(engine)


@contextmanager
def conexion(engine: Engine) -> Iterator:
    """Context manager que da una conexion y la cierra al salir."""
    with engine.connect() as conn:
        yield conn


def probar_conexion(engine: Engine) -> dict:
    """Ping a la base. Devuelve dict con info o levanta la excepcion de SQLAlchemy.

    Util para un `make check-db` o para el endpoint `/health` de una API real.
    """
    with engine.connect() as conn:
        resultado = conn.execute(text("SELECT 1")).scalar()
    return {
        "ok": resultado == 1,
        "url": database_url_safe(str(engine.url)),
        "driver": engine.dialect.name,
    }

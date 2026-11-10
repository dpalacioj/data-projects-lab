"""Tests de la capa de BD: engine, conexion, DDL, PRAGMAs."""
from ingesta_eventos.db import probar_conexion
from sqlalchemy import inspect


def test_probar_conexion_responde(engine_memoria):
    info = probar_conexion(engine_memoria)
    assert info["ok"] is True
    assert info["driver"] == "sqlite"


def test_crear_tablas_declara_las_cuatro(engine_memoria):
    inspector = inspect(engine_memoria)
    tablas = set(inspector.get_table_names())
    assert {"clientes", "productos", "pedidos", "items_pedido"}.issubset(tablas)


def test_database_url_safe_oculta_password():
    from ingesta_eventos.config import database_url_safe

    url = "postgresql+psycopg://admin:mi_clave@localhost:5432/mi_db"
    oculta = database_url_safe(url)
    assert "mi_clave" not in oculta
    assert "***" in oculta
    assert "admin" in oculta
    assert "localhost:5432/mi_db" in oculta


def test_database_url_safe_no_rompe_con_sqlite():
    from ingesta_eventos.config import database_url_safe

    url = "sqlite:///data/db/eventos.db"
    assert database_url_safe(url) == url  # sin password, se devuelve igual


def test_build_url_cae_a_sqlite_por_defecto(monkeypatch):
    """Sin DATABASE_URL ni credenciales de Postgres -> SQLite local."""
    for var in ("DATABASE_URL", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_NAME"):
        monkeypatch.delenv(var, raising=False)

    from importlib import reload

    from ingesta_eventos import config

    reload(config)
    assert config.DATABASE_URL.startswith("sqlite:///")


def test_build_url_arma_postgres_desde_variables(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_USER", "app")
    monkeypatch.setenv("DB_PASSWORD", "secreto")
    monkeypatch.setenv("DB_HOST", "db.interno")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "eventos_prod")

    from importlib import reload

    from ingesta_eventos import config

    reload(config)
    assert config.DATABASE_URL == (
        "postgresql+psycopg://app:secreto@db.interno:5433/eventos_prod"
    )

    # Limpieza: recarga de nuevo sin esas vars para no contaminar otros tests.
    for var in ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"):
        monkeypatch.delenv(var, raising=False)
    reload(config)

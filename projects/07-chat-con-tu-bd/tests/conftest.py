"""Fixtures compartidas por los tests del proyecto 07.

Idea central: los tests NO deben necesitar Ollama corriendo ni acceso a red.
Usamos:

- `FakeListChatModel` de LangChain: devuelve respuestas predeterminadas, en orden.
- Una BD SQLite en memoria con un mini-schema de ejemplo para que las queries
  generadas por el "LLM" de verdad corran.
"""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from sqlalchemy import create_engine, text


@pytest.fixture(autouse=True)
def _bd_en_memoria(monkeypatch, tmp_path):
    """Redirige `DATABASE_URL` a una SQLite en memoria con datos de prueba.

    `autouse=True` -> aplica a TODOS los tests. Sin esto, `sql_tools.crear_engine`
    intentaria abrir el SQLite del proyecto 06, que puede no existir en CI.

    `tmp_path` para que cada test tenga su propia base (fichero) y no colisionen.
    """
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"

    engine = create_engine(url, future=True)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE productos (id INTEGER PRIMARY KEY, nombre TEXT, precio REAL)"
        ))
        conn.execute(text(
            "CREATE TABLE ventas (id INTEGER PRIMARY KEY, producto_id INTEGER, cantidad INTEGER)"
        ))
        for i in range(1, 6):
            conn.execute(
                text("INSERT INTO productos VALUES (:i, :n, :p)"),
                {"i": i, "n": f"producto_{i}", "p": 10.0 * i},
            )
            conn.execute(
                text("INSERT INTO ventas VALUES (:i, :pid, :c)"),
                {"i": i, "pid": i, "c": i * 2},
            )

    # Monkeypatch de la constante y del cache
    monkeypatch.setenv("DATABASE_URL", url)
    from chat_bd import config, sql_tools
    config.DATABASE_URL = url
    sql_tools.DATABASE_URL = url
    sql_tools.crear_engine.cache_clear()

    yield

    sql_tools.crear_engine.cache_clear()


@pytest.fixture
def llm_fake():
    """Factory para construir LLMs falsos con las respuestas que quieras."""

    def factory(respuestas: list[str]) -> FakeListChatModel:
        return FakeListChatModel(responses=respuestas)

    return factory

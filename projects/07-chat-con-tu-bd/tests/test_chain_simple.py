"""Tests de la cadena LangChain simple con FakeListChatModel."""
from chat_bd.chain_simple import ejecutar


def test_chain_completo_con_llm_fake(llm_fake):
    """El LLM devuelve primero el SQL y luego la respuesta natural."""
    llm = llm_fake(
        [
            "SELECT COUNT(*) AS total FROM productos",            # respuesta 1: SQL
            "Hay 5 productos en la base de datos.",               # respuesta 2: NL
        ]
    )
    r = ejecutar("cuantos productos hay?", llm=llm)

    assert r.sql.startswith("SELECT")
    assert r.resultado.iloc[0]["total"] == 5
    assert "5" in r.respuesta


def test_chain_quita_markdown_del_sql(llm_fake):
    """Si el LLM envuelve el SQL en ```sql ... ```, lo desenvolvemos."""
    llm = llm_fake(
        [
            "```sql\nSELECT COUNT(*) AS total FROM productos\n```",
            "Son 5.",
        ]
    )
    r = ejecutar("cuantos?", llm=llm)
    assert "```" not in r.sql
    assert r.sql.startswith("SELECT")


def test_chain_falla_si_el_sql_es_escritura(llm_fake):
    """Si el LLM se le ocurre proponer un DELETE, el ejecutor lo rechaza."""
    import pytest
    from chat_bd.sql_tools import SQLNoPermitidoError

    llm = llm_fake(["DELETE FROM productos", "lo borre todo jeje"])
    with pytest.raises(SQLNoPermitidoError):
        ejecutar("borra todo", llm=llm)

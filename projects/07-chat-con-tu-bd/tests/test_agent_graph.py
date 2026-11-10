"""Tests del agente LangGraph: loop de auto-correccion y limite de reintentos."""
from chat_bd.agent_graph import ejecutar


def test_grafo_feliz_un_solo_intento(llm_fake):
    """SQL valido a la primera: no reintenta."""
    llm = llm_fake(
        [
            "SELECT COUNT(*) AS total FROM productos",
            "Hay 5.",
        ]
    )
    r = ejecutar("cuantos productos?", llm=llm, max_intentos=3)

    assert r.exito is True
    assert r.intentos == 1
    assert "5" in r.respuesta


def test_grafo_reintenta_cuando_falla_el_sql(llm_fake):
    """Primer SQL malo -> error -> regenera -> OK."""
    llm = llm_fake(
        [
            "SELECT * FROM tabla_que_no_existe",    # intento 1: falla
            "SELECT COUNT(*) AS total FROM productos",  # intento 2: funciona
            "Son 5.",                                # respuesta NL
        ]
    )
    r = ejecutar("cuantos productos?", llm=llm, max_intentos=3)

    assert r.exito is True
    assert r.intentos == 2
    assert r.resultado is not None


def test_grafo_se_rinde_tras_max_intentos(llm_fake):
    """Todos los intentos fallan -> se rinde con mensaje claro."""
    llm = llm_fake(
        [
            "SELECT * FROM tabla_fake_1",
            "SELECT * FROM tabla_fake_2",
            "SELECT * FROM tabla_fake_3",
        ]
    )
    r = ejecutar("pregunta imposible", llm=llm, max_intentos=3)

    assert r.exito is False
    assert r.intentos == 3
    assert "no pude responder" in r.respuesta.lower()


def test_grafo_rechaza_sql_no_lectura_y_reintenta(llm_fake):
    """Si el LLM propone DELETE, el validador falla, el grafo reintenta."""
    llm = llm_fake(
        [
            "DELETE FROM productos",                 # intento 1: rechazado
            "SELECT COUNT(*) AS total FROM productos",  # intento 2: OK
            "5 productos.",
        ]
    )
    r = ejecutar("cuantos hay?", llm=llm, max_intentos=3)

    assert r.exito is True
    assert r.intentos == 2

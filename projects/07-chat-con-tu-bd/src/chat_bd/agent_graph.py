"""Agente LangGraph con auto-correccion de SQL.

Esta es la diferencia clave con `chain_simple`: aqui cuando el SQL falla
(sintaxis invalida, tabla que no existe, columna equivocada) el grafo
alimenta el error de vuelta al LLM y le pide reintentar.

Grafo:

    generar_sql -> ejecutar_sql --(ok)--> formatear_respuesta -> END
                      |
                      +--(error, intentos < max)--> generar_sql_con_feedback -> ejecutar_sql
                      |
                      +--(error, intentos >= max)--> formatear_fallo -> END

LangGraph aporta sobre LangChain:
- Estado explicito compartido entre nodos (`AgenteState`).
- Condicionales en las aristas (routing basado en el estado).
- Loops controlados con limite de iteraciones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from .config import AGENTE_MAX_REINTENTOS
from .llm import obtener_llm
from .prompts import prompt_respuesta, prompt_sql, prompt_sql_retry
from .sql_tools import SQLNoPermitidoError, ejecutar_sql_lectura, schema_como_texto


# ---------------------------------------------------------------------------
# Estado compartido
# ---------------------------------------------------------------------------
class AgenteState(TypedDict, total=False):
    pregunta: str
    schema: str
    sql: str
    resultado: pd.DataFrame
    respuesta: str
    error: str | None
    intentos: int
    max_intentos: int


@dataclass
class ResultadoAgente:
    pregunta: str
    sql: str
    resultado: pd.DataFrame | None
    respuesta: str
    intentos: int
    exito: bool
    error: str | None = None
    trazas: list[dict[str, Any]] = field(default_factory=list)


def _limpiar_sql(texto: str) -> str:
    t = texto.strip()
    if t.startswith("```"):
        lineas = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        t = "\n".join(lineas).strip()
    return t


# ---------------------------------------------------------------------------
# Nodos del grafo (cada uno recibe el estado y devuelve un parche al estado)
# ---------------------------------------------------------------------------
def _nodo_generar_sql(llm: BaseChatModel):
    cadena = prompt_sql | llm | StrOutputParser()

    def generar(state: AgenteState) -> dict:
        sql = cadena.invoke({"pregunta": state["pregunta"], "schema": state["schema"]})
        return {"sql": _limpiar_sql(sql), "intentos": state.get("intentos", 0) + 1}

    return generar


def _nodo_generar_sql_con_feedback(llm: BaseChatModel):
    cadena = prompt_sql_retry | llm | StrOutputParser()

    def regenerar(state: AgenteState) -> dict:
        sql = cadena.invoke(
            {
                "pregunta": state["pregunta"],
                "schema": state["schema"],
                "sql_previo": state["sql"],
                "error": state.get("error", ""),
            }
        )
        return {
            "sql": _limpiar_sql(sql),
            "intentos": state.get("intentos", 0) + 1,
            "error": None,
        }

    return regenerar


def _nodo_ejecutar_sql(state: AgenteState) -> dict:
    try:
        df = ejecutar_sql_lectura(state["sql"])
        return {"resultado": df, "error": None}
    except (SQLNoPermitidoError, Exception) as exc:  # noqa: BLE001
        return {"resultado": None, "error": str(exc)}


def _nodo_formatear_respuesta(llm: BaseChatModel):
    cadena = prompt_respuesta | llm | StrOutputParser()

    def formatear(state: AgenteState) -> dict:
        df = state.get("resultado")
        resultado_str = df.head(20).to_string(index=False) if df is not None else "(vacio)"
        respuesta = cadena.invoke(
            {
                "pregunta": state["pregunta"],
                "sql": state["sql"],
                "resultado": resultado_str,
            }
        )
        return {"respuesta": respuesta.strip()}

    return formatear


def _nodo_formatear_fallo(state: AgenteState) -> dict:
    msg = (
        f"No pude responder tu pregunta despues de {state.get('intentos', 0)} intentos. "
        f"Ultimo error: {state.get('error', 'desconocido')}"
    )
    return {"respuesta": msg}


# ---------------------------------------------------------------------------
# Routing condicional
# ---------------------------------------------------------------------------
def _decidir_siguiente(state: AgenteState) -> Literal["retry", "ok", "giveup"]:
    if state.get("error") is None:
        return "ok"
    if state.get("intentos", 0) >= state.get("max_intentos", AGENTE_MAX_REINTENTOS):
        return "giveup"
    return "retry"


# ---------------------------------------------------------------------------
# Construccion del grafo
# ---------------------------------------------------------------------------
def construir_grafo(llm: BaseChatModel | None = None):
    """Ensambla y compila el grafo LangGraph. Devuelve un `Runnable`."""
    llm = llm or obtener_llm()

    grafo = StateGraph(AgenteState)
    grafo.add_node("generar_sql", _nodo_generar_sql(llm))
    grafo.add_node("ejecutar_sql", _nodo_ejecutar_sql)
    grafo.add_node("regenerar_sql", _nodo_generar_sql_con_feedback(llm))
    grafo.add_node("formatear_respuesta", _nodo_formatear_respuesta(llm))
    grafo.add_node("formatear_fallo", _nodo_formatear_fallo)

    grafo.set_entry_point("generar_sql")
    grafo.add_edge("generar_sql", "ejecutar_sql")
    grafo.add_conditional_edges(
        "ejecutar_sql",
        _decidir_siguiente,
        {
            "ok": "formatear_respuesta",
            "retry": "regenerar_sql",
            "giveup": "formatear_fallo",
        },
    )
    grafo.add_edge("regenerar_sql", "ejecutar_sql")
    grafo.add_edge("formatear_respuesta", END)
    grafo.add_edge("formatear_fallo", END)

    return grafo.compile()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def ejecutar(
    pregunta: str,
    llm: BaseChatModel | None = None,
    max_intentos: int = AGENTE_MAX_REINTENTOS,
) -> ResultadoAgente:
    app = construir_grafo(llm)
    estado_inicial: AgenteState = {
        "pregunta": pregunta,
        "schema": schema_como_texto(),
        "intentos": 0,
        "max_intentos": max_intentos,
    }
    final = app.invoke(estado_inicial)

    return ResultadoAgente(
        pregunta=pregunta,
        sql=final.get("sql", ""),
        resultado=final.get("resultado"),
        respuesta=final.get("respuesta", ""),
        intentos=final.get("intentos", 0),
        exito=final.get("error") is None,
        error=final.get("error"),
    )


def responder(pregunta: str) -> str:
    return ejecutar(pregunta).respuesta

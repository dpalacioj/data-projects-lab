"""Cadena LangChain minima: pregunta -> SQL -> ejecuta -> responde.

Es la version "sin agente": una sola pasada, sin reintentos. Muestra lo
basico de LangChain: componer un LLM con prompt y con una funcion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .llm import obtener_llm
from .prompts import prompt_respuesta, prompt_sql
from .sql_tools import ejecutar_sql_lectura, schema_como_texto


@dataclass
class ResultadoChain:
    pregunta: str
    sql: str
    resultado: pd.DataFrame
    respuesta: str
    detalles: dict[str, Any] = field(default_factory=dict)


def _limpiar_sql(texto: str) -> str:
    """El LLM a veces devuelve el SQL envuelto en ```sql ... ```. Lo quitamos."""
    t = texto.strip()
    if t.startswith("```"):
        lineas = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        t = "\n".join(lineas).strip()
    return t


def ejecutar(
    pregunta: str,
    llm: BaseChatModel | None = None,
) -> ResultadoChain:
    """Ejecuta la cadena completa. Permite inyectar un LLM (util para tests)."""
    llm = llm or obtener_llm()
    schema = schema_como_texto()

    # Paso 1: pregunta + schema -> SQL
    cadena_sql = prompt_sql | llm | StrOutputParser()
    sql_crudo = cadena_sql.invoke({"pregunta": pregunta, "schema": schema})
    sql = _limpiar_sql(sql_crudo)

    # Paso 2: ejecuta SQL contra la BD
    df = ejecutar_sql_lectura(sql)

    # Paso 3: SQL + resultado -> respuesta en lenguaje natural
    cadena_respuesta = prompt_respuesta | llm | StrOutputParser()
    respuesta = cadena_respuesta.invoke(
        {
            "pregunta": pregunta,
            "sql": sql,
            "resultado": df.head(20).to_string(index=False),
        }
    )

    return ResultadoChain(
        pregunta=pregunta,
        sql=sql,
        resultado=df,
        respuesta=respuesta.strip(),
        detalles={"filas": len(df)},
    )


def responder(pregunta: str) -> str:
    """Atajo que devuelve solo la respuesta en texto (para el Makefile y CLI)."""
    return ejecutar(pregunta).respuesta

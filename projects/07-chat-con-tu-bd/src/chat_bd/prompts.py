"""Plantillas de prompt centralizadas. Un solo sitio para afinar el tono."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SISTEMA_SQL = """Eres un analista de datos experto en SQL de SQLite.
Tu tarea es convertir preguntas en espanol a una consulta SQL correcta y eficiente.

Reglas estrictas:
1. SOLO devuelve la consulta SQL. Sin explicacion, sin texto adicional, sin markdown.
2. SOLO usa SELECT o WITH. Nunca INSERT, UPDATE, DELETE, DROP o cualquier DDL.
3. Si la pregunta no se puede responder con los datos disponibles, responde:
   SELECT 'pregunta_fuera_de_alcance' AS mensaje;
4. Usa aliases claros (ej: `AS ingresos` en vez de `AS c1`).
5. Siempre incluye LIMIT 50 si no hay agregacion.

Schema de la base disponible:
{schema}
"""

SISTEMA_RESPUESTA = """Eres un analista de datos. Acabas de ejecutar una consulta
SQL para responder una pregunta del usuario. Explica el resultado en lenguaje
natural, en espanol, en 1-3 frases. Usa los numeros exactos del resultado.
Si el resultado esta vacio, dilo claramente.
"""

prompt_sql = ChatPromptTemplate.from_messages(
    [
        ("system", SISTEMA_SQL),
        ("human", "Pregunta: {pregunta}"),
    ]
)

prompt_respuesta = ChatPromptTemplate.from_messages(
    [
        ("system", SISTEMA_RESPUESTA),
        (
            "human",
            "Pregunta original: {pregunta}\n\nSQL ejecutado:\n{sql}\n\n"
            "Resultado (primeras filas):\n{resultado}\n\n"
            "Da la respuesta al usuario.",
        ),
    ]
)

# Prompt para auto-correccion en el grafo: el LLM recibe el error y rehace SQL.
prompt_sql_retry = ChatPromptTemplate.from_messages(
    [
        ("system", SISTEMA_SQL),
        ("human", "Pregunta: {pregunta}"),
        (
            "human",
            "Tu intento anterior fallo. SQL que probaste:\n{sql_previo}\n\n"
            "Error: {error}\n\nCorrige el SQL y devuelve SOLO la nueva consulta.",
        ),
    ]
)

"""Utilidades de BD: inspeccion del schema y ejecucion SEGURA de SQL.

El LLM genera SQL. Correr ese SQL contra la base sin filtros es un riesgo:
si el LLM decide "DROP TABLE clientes" lo hace. Aqui ponemos las barreras:

- `ejecutar_sql_lectura`: solo permite SELECT y rechaza todo lo demas.
- `resumen_bd`: expone el schema al LLM como texto (lo que necesita para
  escribir SQL correcto).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .config import DATABASE_URL

# Palabras que convierten un SQL en escritura/DDL. Las bloqueamos de raiz.
VERBOS_PROHIBIDOS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "ATTACH", "DETACH",
    "VACUUM", "PRAGMA", "MERGE",
}


class SQLNoPermitidoError(Exception):
    """Se intento ejecutar un SQL que no es de solo lectura."""


@lru_cache(maxsize=1)
def crear_engine() -> Engine:
    """Engine cacheado: una sola conexion durante la vida del proceso."""
    return create_engine(DATABASE_URL, future=True)


def _normalizar(sql: str) -> str:
    """Quita comentarios y normaliza espacios para analizar."""
    # Quita comentarios de linea y de bloque.
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip().rstrip(";").strip()


def validar_sql_lectura(sql: str) -> str:
    """Valida que el SQL sea una sola sentencia SELECT (o WITH ... SELECT).

    Devuelve el SQL limpio si pasa; levanta `SQLNoPermitidoError` si no.
    """
    limpio = _normalizar(sql)
    if not limpio:
        raise SQLNoPermitidoError("SQL vacio")

    # Prohibimos multi-sentencia (';' en medio tras normalizar).
    if ";" in limpio:
        raise SQLNoPermitidoError("Solo se acepta una sentencia por consulta")

    # Primera palabra debe ser SELECT o WITH (CTE).
    primera = limpio.split(None, 1)[0].upper()
    if primera not in {"SELECT", "WITH"}:
        raise SQLNoPermitidoError(
            f"Solo se permite SELECT/WITH. Verbo detectado: {primera}"
        )

    # Verbo prohibido en cualquier parte (busqueda por palabra).
    for verbo in VERBOS_PROHIBIDOS:
        if re.search(rf"\b{verbo}\b", limpio, flags=re.IGNORECASE):
            raise SQLNoPermitidoError(
                f"Verbo prohibido detectado: {verbo}. Solo lecturas."
            )

    return limpio


def ejecutar_sql_lectura(sql: str, limite: int = 500) -> pd.DataFrame:
    """Ejecuta un SELECT y devuelve un DataFrame. Aplica LIMIT si no hay.

    Si el SQL ya trae LIMIT, lo respeta. Si no, inyecta uno al final para
    evitar que una consulta accidental devuelva un millon de filas al LLM.
    """
    limpio = validar_sql_lectura(sql)

    if not re.search(r"\bLIMIT\b", limpio, flags=re.IGNORECASE):
        limpio = f"{limpio} LIMIT {limite}"

    engine = crear_engine()
    return pd.read_sql(text(limpio), engine)


def resumen_bd() -> dict[str, Any]:
    """Resumen del schema para el LLM: tablas, columnas, 3 filas de muestra."""
    engine = crear_engine()
    inspector = inspect(engine)

    out: dict[str, Any] = {"tablas": {}, "url": _url_oculta(DATABASE_URL)}
    for nombre in sorted(inspector.get_table_names()):
        cols = [
            {"nombre": c["name"], "tipo": str(c["type"])}
            for c in inspector.get_columns(nombre)
        ]
        with engine.connect() as conn:
            total = int(conn.execute(text(f"SELECT COUNT(*) FROM {nombre}")).scalar() or 0)
            muestra = pd.read_sql(text(f"SELECT * FROM {nombre} LIMIT 3"), conn)
        out["tablas"][nombre] = {
            "columnas": cols,
            "n_filas": total,
            "muestra": muestra.to_dict(orient="records"),
        }
    return out


def schema_como_texto() -> str:
    """Schema en formato compacto para inyectar en el prompt del LLM."""
    resumen = resumen_bd()
    lineas = []
    for tabla, info in resumen["tablas"].items():
        cols = ", ".join(f"{c['nombre']} {c['tipo']}" for c in info["columnas"])
        lineas.append(f"- {tabla}({cols})  -- {info['n_filas']:,} filas")
    return "\n".join(lineas)


def _url_oculta(url: str) -> str:
    """Enmascara password al mostrar la URL (patron del proyecto 06)."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    userinfo, hostinfo = rest.split("@", 1)
    if ":" in userinfo:
        user, _ = userinfo.split(":", 1)
        userinfo = f"{user}:***"
    return f"{scheme}://{userinfo}@{hostinfo}"

"""Lectura de los archivos JSON crudos de `data/raw/`.

Esta capa NO transforma. Solo lee bytes, los parsea y los valida contra el
schema de Pydantic. Si algo viene mal formado, aqui explota (fail fast).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .config import RAW_DIR
from .schemas import LoteJSON


def listar_lotes(directorio: Path = RAW_DIR, patron: str = "eventos_*.json") -> list[Path]:
    """Devuelve la lista ordenada de archivos JSON a procesar."""
    return sorted(directorio.glob(patron))


def leer_lote(path: Path) -> LoteJSON:
    """Lee un archivo JSON y lo valida contra el schema del lote.

    Si el JSON no cumple el esquema, levanta `pydantic.ValidationError` con
    la ubicacion exacta del problema.
    """
    with path.open("r", encoding="utf-8") as f:
        crudo = json.load(f)
    try:
        return LoteJSON.model_validate(crudo)
    except ValidationError as exc:
        raise ValueError(f"JSON invalido en {path.name}: {exc}") from exc


def iterar_lotes(directorio: Path = RAW_DIR) -> Iterator[LoteJSON]:
    """Generator sobre todos los archivos del directorio. No carga todo en RAM."""
    for path in listar_lotes(directorio):
        yield leer_lote(path)

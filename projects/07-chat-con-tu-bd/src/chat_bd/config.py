"""Configuracion centralizada: proveedor de LLM, modelo, URL de BD."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _resolver_database_url() -> str:
    """Resuelve la URL de BD a path absoluto si es SQLite relativo.

    El .env por defecto apunta a `../06-ingesta-eventos-ecommerce/data/db/eventos.db`,
    relativo a este proyecto. SQLAlchemy necesita path absoluto para que funcione
    sin importar desde donde se invoque.
    """
    url = os.getenv(
        "DATABASE_URL",
        "sqlite:///../06-ingesta-eventos-ecommerce/data/db/eventos.db",
    )
    prefijo = "sqlite:///"
    if url.startswith(prefijo):
        path = url[len(prefijo):]
        if not Path(path).is_absolute():
            abs_path = (PROJECT_DIR / path).resolve()
            return f"{prefijo}{abs_path}"
    return url


LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_TEMPERATURA: float = float(os.getenv("LLM_TEMPERATURA", "0.0"))

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")
HF_MODEL: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

DATABASE_URL: str = _resolver_database_url()

AGENTE_MAX_REINTENTOS: int = int(os.getenv("AGENTE_MAX_REINTENTOS", "3"))

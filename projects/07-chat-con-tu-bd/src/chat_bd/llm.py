"""Factory de ChatModel. Un unico lugar que decide que LLM usar.

Soporta 3 proveedores:

- `ollama` -> `ChatOllama`  (local, open source, default).
- `hf`     -> `HuggingFaceEndpoint` (API remota de HF; requiere HF_TOKEN).
- `fake`   -> `FakeListChatModel` (tests; devuelve respuestas fijas).

El resto del paquete llama `obtener_llm()` y no sabe nada del proveedor.
"""
from __future__ import annotations

from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from .config import (
    HF_MODEL,
    HF_TOKEN,
    LLM_PROVIDER,
    LLM_TEMPERATURA,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def obtener_llm(
    proveedor: str | None = None,
    respuestas_fake: Sequence[str] | None = None,
) -> BaseChatModel:
    """Devuelve una instancia del ChatModel segun el proveedor configurado.

    Argumentos:
        proveedor: fuerza un proveedor especifico (default: el del .env).
        respuestas_fake: lista de respuestas si `proveedor="fake"`.
    """
    proveedor = (proveedor or LLM_PROVIDER).lower()

    if proveedor == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURA,
        )

    if proveedor == "hf":
        if not HF_TOKEN:
            raise ValueError(
                "LLM_PROVIDER=hf requiere HF_TOKEN. Creas uno gratis en "
                "https://huggingface.co/settings/tokens"
            )
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=LLM_TEMPERATURA or 0.01,  # HF no acepta exactamente 0
            task="text-generation",
        )
        return ChatHuggingFace(llm=endpoint)

    if proveedor == "fake":
        if not respuestas_fake:
            raise ValueError("fake LLM requiere `respuestas_fake` (lista de str)")
        return FakeListChatModel(responses=list(respuestas_fake))

    raise ValueError(
        f"proveedor desconocido: {proveedor!r}. Usa 'ollama', 'hf' o 'fake'."
    )

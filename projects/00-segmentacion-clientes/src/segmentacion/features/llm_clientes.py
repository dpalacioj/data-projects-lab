"""Adaptadores de cliente LLM para `features.texto_llm`.

- `FakeLLM(ruta)`: cliente por defecto en cuadernos y pruebas; responde desde
  un JSON de respuestas canonicas. **Sin red.**
- `cliente_openai_compatible(base_url, api_key_env, modelo, temperature=0.0)`:
  unico adaptador real. Cubre cualquier servicio con API compatible con el
  formato de chat de OpenAI (DeepSeek, OpenAI, Groq, Ollama local, etc.).
  La clave se lee de la variable de entorno `api_key_env`; nunca aparece en
  codigo ni en el repositorio.

Ningun archivo del repositorio (fuera de este adaptador y del README) nombra
un proveedor; el prompt es neutro, los tests usan `FakeLLM` y el cuaderno 08
tiene dos lineas comentadas para pasar al cliente real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class FakeLLM:
    """Cliente determinista que responde desde `ruta` (JSON).

    Formato esperado del JSON:

        {
            "<prompt_substring>": "<respuesta como arreglo JSON en string>",
            ...
        }

    En tiempo de llamada, se busca cualquier substring del prompt que
    aparezca como clave en el diccionario y se devuelve su valor. Si
    ninguna clave hace match, devuelve `"[]"` (arreglo vacio).
    """

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self._respuestas: dict[str, str] = json.loads(self.ruta.read_text())

    def __call__(self, prompt: str) -> str:
        for substring, respuesta in self._respuestas.items():
            if substring in prompt:
                return respuesta
        return "[]"


def cliente_openai_compatible(
    base_url: str,
    api_key_env: str,
    modelo: str,
    temperature: float = 0.0,
):
    """Devuelve un `ClienteLLM` que llama a cualquier endpoint OpenAI-compatible.

    - `base_url`: endpoint del servicio.
    - `api_key_env`: nombre de la variable de entorno con la clave API.
      La clave nunca se escribe en codigo ni en el repositorio.
    - `modelo`: identificador del modelo a invocar.
    - `temperature`: 0.0 por defecto (determinismo).

    Lanza `RuntimeError` si `api_key_env` no esta definida en el entorno.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"La variable de entorno `{api_key_env}` no esta definida. "
            "Configura `.env` (ver `.env.example`) antes de usar el cliente real."
        )

    # Import diferido: no se importa `requests` al cargar el modulo para no
    # añadirlo como dependencia obligatoria.
    import urllib.request

    def _llamar(prompt: str) -> str:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            data=json.dumps(
                {
                    "model": modelo,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
            ).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]

    return _llamar

"""API sencilla con FastAPI: la tercera interfaz del mismo Predictor (Unidad 4).

Correr desde la carpeta del proyecto:

    uv run uvicorn api.main:app --reload       -> http://127.0.0.1:8000/docs

Mismo patron que projects/05-ejemplo-completo/app/api_fastapi.py, con el
modelo cargado una sola vez en `lifespan`.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentacion.servicio import EntradaCliente, Predictor  # noqa: E402

recursos: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    recursos["predictor"] = Predictor.desde_carpeta()  # al arrancar, una sola vez
    yield
    recursos.clear()


app = FastAPI(title="Recompra · API", version="1.0", lifespan=lifespan)


@app.get("/salud")
def salud() -> dict:
    return recursos["predictor"].salud()


@app.post("/predecir")
def predecir(entrada: EntradaCliente, umbral: float | None = None) -> dict:
    return recursos["predictor"].predecir(entrada, umbral=umbral)

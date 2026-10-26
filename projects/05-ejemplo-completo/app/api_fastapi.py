"""API FastAPI: endpoint POST /predict con JSON de entrada y salida.

Correr desde la carpeta del proyecto:
    uv run uvicorn app.api_fastapi:app --reload

O con make:
    make serve-api

Probar:
    curl -X POST -H "Content-Type: application/json" \\
        -d '{"alcohol": 13.5, "malic_acid": 2.0, ... }' \\
        http://localhost:8000/predict
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ejemplo_completo.config import CLASES, FEATURES_NUM, MODEL_PATH  # noqa: E402
from ejemplo_completo.predict import predecir  # noqa: E402

app = FastAPI(
    title="Clasificador de vinos (Palmer)",
    version="0.1.0",
    description="Clasifica cultivares de vino sobre el dataset load_wine de sklearn.",
)


class EntradaMuestra(BaseModel):
    """Caracteristicas de una muestra de vino (todas las columnas numericas)."""

    alcohol: float = Field(..., gt=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., alias="od280/od315_of_diluted_wines", ge=0)
    proline: float = Field(..., ge=0)

    model_config = {"populate_by_name": True}


class SalidaPrediccion(BaseModel):
    clase: str
    probabilidades: dict[str, float]


@app.get("/")
def info():
    return {
        "proyecto": "Clasificador de vinos",
        "clases": CLASES,
        "features_esperadas": FEATURES_NUM,
        "endpoints": {"POST /predict": "clasifica una muestra"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": MODEL_PATH.exists()}


@app.post("/predict", response_model=SalidaPrediccion)
def predict(entrada: EntradaMuestra):
    datos = entrada.model_dump(by_alias=True)
    try:
        resultado = predecir(datos)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return resultado

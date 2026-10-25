"""API FastAPI: endpoint POST /predict que recibe una imagen y devuelve top-3.

Correr:
    uv run uvicorn projects.reconocimiento_imagenes.app.api_fastapi:app --reload

Probar:
    curl -X POST -F "file=@mi_imagen.jpg" http://localhost:8000/predict
"""
import io
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconocimiento_imagenes.config import CLASES, MODEL_PATH  # noqa: E402
from reconocimiento_imagenes.predict import cargar_modelo, predecir  # noqa: E402

app = FastAPI(title="Reconocimiento de prendas", version="0.1.0")

_modelo = None


def obtener_modelo():
    """Carga lazy; evita cargar el modelo al importar el modulo (p. ej. en tests)."""
    global _modelo
    if _modelo is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo no encontrado en {MODEL_PATH}. "
                    "Entrena con: uv run python -m reconocimiento_imagenes.train"
                ),
            )
        _modelo = cargar_modelo()
    return _modelo


@app.get("/")
def info():
    return {
        "proyecto": "Reconocimiento de prendas (Fashion-MNIST)",
        "clases": CLASES,
        "endpoints": {"POST /predict": "subir imagen, recibir top-3 predicciones"},
    }


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se esperaba una imagen.")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {exc}") from exc

    modelo = obtener_modelo()
    predicciones = predecir(modelo, img, topk=3)
    return {
        "nombre_archivo": file.filename,
        "predicciones": [{"clase": c, "probabilidad": p} for c, p in predicciones],
    }

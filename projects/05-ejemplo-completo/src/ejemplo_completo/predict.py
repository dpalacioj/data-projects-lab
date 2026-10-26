"""Inferencia: carga del modelo entrenado y prediccion sobre una muestra."""
import joblib
import pandas as pd

from .config import CLASES, FEATURES_NUM, MODEL_PATH

_pipeline = None


def obtener_pipeline():
    """Carga perezosa del modelo. Lo guarda en memoria para llamadas siguientes."""
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {MODEL_PATH}. "
                "Entrena primero con: `uv run python -m ejemplo_completo.train` "
                "o `make train` desde la carpeta del proyecto."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predecir(datos: dict) -> dict:
    """Dado un dict con una muestra, devuelve clase predicha y probabilidades.

    Ejemplo de entrada:
        {"alcohol": 13.5, "malic_acid": 2.0, ... }  (todas las claves de FEATURES_NUM)

    Ejemplo de salida:
        {
            "clase": "cultivar_1",
            "probabilidades": {"cultivar_0": 0.1, "cultivar_1": 0.8, "cultivar_2": 0.1}
        }
    """
    faltantes = [c for c in FEATURES_NUM if c not in datos]
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    pipeline = obtener_pipeline()
    df = pd.DataFrame([datos], columns=FEATURES_NUM)
    pred_idx = int(pipeline.predict(df)[0])
    probs = pipeline.predict_proba(df)[0]

    return {
        "clase": CLASES[pred_idx],
        "probabilidades": {CLASES[i]: float(p) for i, p in enumerate(probs)},
    }

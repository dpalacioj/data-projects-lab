"""Carga del modelo entrenado y prediccion sobre una imagen PIL."""
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .config import CLASES, IMG_SIZE, MODEL_PATH
from .data import TRANSFORM
from .model import SimpleCNN


def cargar_modelo(ruta: Path = MODEL_PATH, device: str | None = None) -> SimpleCNN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = SimpleCNN()
    modelo.load_state_dict(torch.load(ruta, map_location=device))
    modelo.to(device).eval()
    return modelo


def preparar_imagen(img: Image.Image) -> torch.Tensor:
    """Pasa una imagen arbitraria al formato del dataset: 1 canal, 28x28, normalizada."""
    img = ImageOps.grayscale(img)
    img = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE))
    return TRANSFORM(img).unsqueeze(0)


@torch.no_grad()
def predecir(modelo: SimpleCNN, img: Image.Image, topk: int = 3) -> list[tuple[str, float]]:
    """Devuelve las topk clases mas probables como (nombre_clase, probabilidad)."""
    device = next(modelo.parameters()).device
    tensor = preparar_imagen(img).to(device)
    logits = modelo(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    topk_vals, topk_idx = torch.topk(probs, k=topk)
    return [(CLASES[int(i)], float(v)) for v, i in zip(topk_vals, topk_idx)]

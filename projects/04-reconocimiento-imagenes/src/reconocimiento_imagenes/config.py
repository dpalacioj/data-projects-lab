"""Constantes del proyecto. Un solo lugar para cambiar paths, clases e hiperparametros."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "fashion_cnn.pth"

CLASES = [
    "camiseta",
    "pantalon",
    "sueter",
    "vestido",
    "abrigo",
    "sandalia",
    "camisa",
    "zapatilla",
    "bolso",
    "bota",
]

IMG_SIZE = 28
NUM_CLASES = len(CLASES)

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
SEMILLA = 42

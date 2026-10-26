"""Constantes del proyecto. Un solo lugar para cambiar paths e hiperparametros."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
MODEL_PATH = MODELS_DIR / "clasificador_vino.joblib"

CLASES = ["cultivar_0", "cultivar_1", "cultivar_2"]

FEATURES_NUM = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]
TARGET = "cultivar"

SEMILLA = 42
TEST_SIZE = 0.25
N_ESTIMATORS = 200

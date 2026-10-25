"""
Configuración centralizada del proyecto.

Centraliza rutas, nombres de archivos, parámetros y constantes
para evitar duplicación y facilitar el mantenimiento.

Uso típico desde un notebook:
    from segmentacion.config import RAW_DATA_FILE, RFM_FEATURES, RANDOM_STATE
"""

from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# Path(__file__) es la ruta de este archivo (config.py), ubicado en
# src/segmentacion/. Tres .parent suben a: segmentacion/ → src/ → raiz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Archivos de datos
# ---------------------------------------------------------------------------
# Dataset original: Online Retail II (UCI ML Repository, ID 502)
# URL: https://archive.ics.uci.edu/dataset/502/online+retail+ii
RAW_DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DATA_FILE = RAW_DATA_DIR / "online_retail_II.xlsx"

# Datasets generados durante el flujo
CLEAN_DATA_FILE = INTERIM_DATA_DIR / "transacciones_limpias.parquet"
RFM_DATA_FILE = PROCESSED_DATA_DIR / "rfm_clientes.parquet"
FEATURES_DATA_FILE = PROCESSED_DATA_DIR / "features_clientes.parquet"
SEGMENTS_FILE = PROCESSED_DATA_DIR / "clientes_segmentados.parquet"

# Modelos
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.joblib"
DBSCAN_MODEL_FILE = MODELS_DIR / "dbscan_model.joblib"
PIPELINE_FILE = MODELS_DIR / "preprocessing_pipeline.joblib"
SEGMENT_PROFILES_FILE = MODELS_DIR / "segment_profiles.json"

# ---------------------------------------------------------------------------
# Variables / Features
# ---------------------------------------------------------------------------
# Columnas RFM básicas (calculadas a nivel de cliente)
RFM_FEATURES = ["Recency", "Frequency", "Monetary"]

# Features extendidas (incluyen comportamiento adicional)
EXTENDED_NUMERIC_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgTicket",  # ticket promedio por factura
    "ProductDiversity",  # cantidad de productos distintos comprados
    "AvgQuantity",  # cantidad promedio de items por factura
]

# Variables categóricas (si se incluye país, por ejemplo)
CATEGORICAL_FEATURES = ["CountryGroup"]

# ---------------------------------------------------------------------------
# Unidad 3: cohortes de recompra en 90 dias
# ---------------------------------------------------------------------------
# Corte A (2011-06-01): se usa para construir cohortes de ENTRENAMIENTO y validacion
# cruzada. Corte B (2011-09-01): se usa para la cohorte de PRUEBA (hold-out temporal).
# El horizonte es la ventana en la que se mide la etiqueta de recompra tras el corte.
FECHA_CORTE_A = datetime(2011, 6, 1)
FECHA_CORTE_B = datetime(2011, 9, 1)
HORIZONTE = 90
SEMILLA = 42  # unica semilla del paquete; alias historico: RANDOM_STATE

# ---------------------------------------------------------------------------
# Hiperparámetros y semillas (U2 legacy)
# ---------------------------------------------------------------------------
RANDOM_STATE = SEMILLA

# K-Means
KMEANS_DEFAULT_K = 4
KMEANS_K_RANGE = range(2, 11)
KMEANS_PARAMS = {
    "n_init": 10,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}

# DBSCAN
DBSCAN_DEFAULT_EPS = 0.5
DBSCAN_DEFAULT_MIN_SAMPLES = 5
DBSCAN_PARAMS = {
    "metric": "euclidean",
    "n_jobs": -1,
}

# PCA (para visualización 2D)
PCA_COMPONENTS_VIZ = 2

# ---------------------------------------------------------------------------
# Crear directorios si no existen
# ---------------------------------------------------------------------------
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(exist_ok=True, parents=True)

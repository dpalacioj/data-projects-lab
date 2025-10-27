"""
Módulo de Configuración

Este módulo centraliza todas las constantes, rutas y parámetros de configuración
del proyecto para facilitar el mantenimiento y evitar duplicación de código.
"""

from pathlib import Path

# Rutas del proyecto
# Path(__file__).parent obtiene la carpeta actual (src/)
# .parent sube un nivel a la raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivos de datos
RAW_DATA_FILE = RAW_DATA_DIR / "automoviles_usados.parquet"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "automoviles_procesados.parquet"

# Archivo del modelo
MODEL_FILE = MODELS_DIR / "random_forest_best_model.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Características del modelo
# Estas listas definen qué columnas son numéricas y cuáles categóricas
NUMERIC_FEATURES = [
    'año',
    'kilometraje',
    'cilindrada',
    'potencia',
    'peso',
    'consumo',
    'edad_propietarios',
    'calificacion_estado'
]

CATEGORICAL_FEATURES = [
    'marca',
    'tipo_carroceria',
    'tipo_combustible',
    'transmision',
    'color',
    'region_venta'
]

# Variable objetivo
TARGET_VARIABLE = 'precio'

# Parámetros de entrenamiento
RANDOM_STATE = 42  # Semilla para reproducibilidad
TEST_SIZE = 0.2    # Proporción de datos para prueba (20%)

# Parámetros de Random Forest (versión 1 - balanceado)
# Estos son los hiperparámetros por defecto del modelo
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Parámetros de Random Forest (versión 2 - más profundo)
# Un modelo más complejo que puede capturar relaciones más complejas
RF_DEEP_PARAMS = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Parámetros de LightGBM
# LightGBM es un modelo basado en gradient boosting, eficiente y preciso
LGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 15,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# Parámetros de KNN (K-Nearest Neighbors)
# KNN predice basándose en los K vecinos más cercanos
KNN_PARAMS = {
    'n_neighbors': 10,
    'weights': 'distance',  # Peso por distancia (más cercanos tienen más peso)
    'algorithm': 'auto',
    'leaf_size': 30,
    'n_jobs': -1
}

# Configuración de MLFlow
MLFLOW_EXPERIMENT_NAME = "prediccion_precios_automoviles"

# Crear directorios si no existen
# exist_ok=True evita errores si el directorio ya existe
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                 MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

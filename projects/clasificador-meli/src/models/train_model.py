"""
Script principal de entrenamiento del modelo Clasificador MercadoLibre.

Este módulo ejecuta el pipeline completo de entrenamiento:
- Carga de datos (parquet preprocesado o JSON crudo)
- Preprocesamiento y feature engineering (si es necesario)
- Entrenamiento del modelo XGBoost
- Evaluación de métricas
- Guardado del modelo entrenado

Uso:
    python train.py              # Entrenar con hiperparámetros base
    python train.py --optimize   # Entrenar con optimización GridSearchCV
    python train.py --from-json  # Forzar procesamiento desde JSON
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from joblib import dump
import logging
import argparse

from src.config import (
    DATA_PATH_RAW, DATA_PATH_RAW_NAME, DATA_PATH_PROCESSED,
    DATA_PATH_PROCESSED_NAME, BASE_CONFIG_PATH, BASE_CONFIG_XG_NAME,
    BASE_CONFIG_XG_OPTIMIZED_NAME, MODEL_PATH, MODEL_NAME
)
from src.utils import load_model_config, save_model_config
from src.features import Preprocessing
from src.models import XGBoostClassifierModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main(optimize=False, from_json=False):
    """
    Ejecuta el pipeline completo de entrenamiento del modelo.

    Args:
        optimize: Si es True, ejecuta GridSearchCV para optimizar hiperparámetros.
                  Si es False, usa hiperparámetros del archivo base_model_xg_config.json
        from_json: Si es True, fuerza procesamiento desde JSON incluso si existe parquet.
                   Si es False, usa parquet si está disponible.

    Flujo de ejecución:
        1. Intenta cargar datos preprocesados desde .parquet (más rápido)
        2. Si no existe o from_json=True, procesa desde MLA_100k.jsonlines
        3. Divide en train/test
        4. Entrena el modelo XGBoost
        5. Evalúa métricas en conjunto de test
        6. Guarda el modelo entrenado en formato .pkl
        7. Guarda las columnas codificadas
    """
    logger.info("Inicio del pipeline de entrenamiento")

    # Rutas usando pathlib (portables)
    raw_data_file = DATA_PATH_RAW / DATA_PATH_RAW_NAME
    processed_data_file = DATA_PATH_PROCESSED / DATA_PATH_PROCESSED_NAME
    parameters_path = BASE_CONFIG_PATH / BASE_CONFIG_XG_NAME
    model_file = MODEL_PATH / MODEL_NAME

    # Asegurar que directorios existan
    DATA_PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Carga de datos: preferir parquet preprocesado si existe
    if processed_data_file.exists() and not from_json:
        logger.info(f"Cargando datos preprocesados desde {processed_data_file}")
        data_transformed = pd.read_parquet(processed_data_file)
        logger.info(f"Datos cargados: {data_transformed.shape}")
    else:
        if from_json:
            logger.info("Forzando procesamiento desde JSON (--from-json)")
        else:
            logger.info(f"No se encontró {processed_data_file}, procesando desde JSON")

        logger.info(f"Cargando datos crudos desde {raw_data_file}")
        with open(raw_data_file) as f:
            data_loaded = [json.loads(line) for line in f]
        data = pd.DataFrame(data_loaded)
        logger.info(f"Datos crudos cargados: {data.shape}")

        preprocessor = Preprocessing()
        data_transformed = preprocessor.transform(data)
        logger.info(f"Datos procesados: {data_transformed.shape}")

        # Guardar datos procesados para futuras ejecuciones
        data_transformed.to_parquet(processed_data_file, index=False)
        logger.info(f"Datos procesados guardados en {processed_data_file}")

    trainer = XGBoostClassifierModel()
    X_train, X_test, y_train, y_test = trainer.preprocess_data(data_transformed)

    # Cargar o optimizar hiperparámetros
    if optimize:
        logger.info("Ejecutando GridSearchCV para encontrar mejores hiperparámetros...")
        params_model = trainer.optimize_hyperparameters(X_train, y_train)
        logger.info("Mejores parámetros encontrados:")
        for k, v in params_model.items():
            logger.info(f"  {k}: {v}")

        # Guardar hiperparámetros optimizados
        optimized_config_path = BASE_CONFIG_PATH / BASE_CONFIG_XG_OPTIMIZED_NAME
        save_model_config(params_model, str(optimized_config_path))
        logger.info(f"Configuración optimizada guardada en {optimized_config_path}")
    else:
        logger.info(f"Cargando configuración base del modelo desde {parameters_path}")
        params_model = load_model_config(str(parameters_path))

    # Entrenar modelo
    logger.info("Entrenando modelo final")
    trainer.train_model(X_train, y_train, params_model)

    # Evaluar métricas
    metrics, _, _ = trainer.evaluate_model(X_test, y_test)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    # Guardar modelo entrenado
    logger.info("Guardando modelo entrenado")
    dump(trainer.get_model(), model_file)
    logger.info(f"Modelo guardado en {model_file}")
    logger.info("Entrenamiento finalizado con éxito")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--from-json", action="store_true", help="Force processing from JSON instead of parquet")
    args = parser.parse_args()

    main(optimize=args.optimize, from_json=args.from_json)
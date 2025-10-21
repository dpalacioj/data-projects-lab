import os
import pandas as pd
import numpy as np
import json
from joblib import dump
import logging
import argparse
from sklearn.preprocessing import LabelEncoder

from src.config import (DATA_PATH_RAW, DATA_PATH_RAW_NAME, DATA_PATH_PROCESSED, DROP_COLUMNS,
                        DATA_PATH_PROCESSED_NAME, BASE_CONFIG_PATH, BASE_CONFIG_XG_NAME,
                        BASE_CONFIG_XG_OPTIMIZED_NAME, MODEL_PATH, ENCODED_COLUMNS_PATH,
                        SELECTED_COLUMNS
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

def main(optimize = False):
    logger.info("Inicio del pipeline de entrenamiento")
    data_path = DATA_PATH_RAW
    data_processed_path = DATA_PATH_PROCESSED
    data_raw_file = DATA_PATH_RAW_NAME
    parameters_xg_path = BASE_CONFIG_PATH
    parameters_xg_name = BASE_CONFIG_XG_NAME
    model_path = MODEL_PATH
    file_path = os.path.join(data_path, data_raw_file)
    file_processed_path = os.path.join(data_processed_path, 'data_model.csv')
    parameters_path = os.path.join(parameters_xg_path, parameters_xg_name)
    model_name = os.path.join(model_path, 'xgb_model_v1.pkl')

    preprocessor = Preprocessing()

    data_loaded = [json.loads(x) for x in open(file_path)]
    data = pd.DataFrame(data_loaded)
    data_transformed = preprocessor.transform(data)

    trainer = XGBoostClassifierModel()
    X_train, X_test, y_train, y_test = trainer.preprocess_data(data_transformed)

    if optimize:
        logger.info("Ejecutando GridSearchCV para encontrar mejores hiperparámetros...")
        params_model = trainer.optimize_hyperparameters(X_train, y_train)
        logger.info("Mejores parámetros encontrados:")
        for k, v in params_model.items():
            logger.info(f"  {k}: {v}")
        save_model_config(params_model, parameters_xg_path)
        logger.info(f"Configuración optimizada guardada en {parameters_xg_path}")
    else:
        logger.info(f"Cargando configuración base del modelo desde {parameters_path}")
        params_model = load_model_config(parameters_path)

    # print(params_model)

    logger.info("Entrenando modelo final")

    trainer.train_model(X_train, y_train, params_model)

    metrics, _, _ = trainer.evaluate_model(X_test, y_test)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    logger.info("Guardando modelo entrenado")
    dump(trainer.get_model(), model_name)
    logger.info(f"Modelo guardado en {model_path}")

    np.save(ENCODED_COLUMNS_PATH, np.array(SELECTED_COLUMNS))

    logger.info("Entrenamiento finalizado con éxito")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    args = parser.parse_args()

    main(optimize=args.optimize)
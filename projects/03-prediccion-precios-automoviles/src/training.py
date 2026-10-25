"""
Módulo de Entrenamiento

Este módulo contiene funciones para entrenar modelos de machine learning,
evaluar su desempeño y guardar los modelos entrenados.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# LightGBM es opcional - solo se importa si está instalado
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# MLFlow es opcional - solo se importa si se usa
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config import (
    MODEL_FILE,
    MODEL_METADATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    RF_PARAMS,
    RF_DEEP_PARAMS,
    LGBM_PARAMS,
    KNN_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR
)
from preprocessing import (
    load_data,
    validate_data,
    prepare_features,
    create_preprocessor,
    get_feature_names
)


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de evaluación para modelos de regresión.

    Args:
        y_true (array-like): Valores reales
        y_pred (array-like): Valores predichos

    Returns:
        dict: Diccionario con las métricas calculadas

    Métricas:
        - RMSE: Root Mean Squared Error (error cuadrático medio)
               Penaliza fuertemente los errores grandes
        - MAE: Mean Absolute Error (error absoluto medio)
              Más robusto a outliers que RMSE
        - R²: Coeficiente de determinación (0 a 1)
             Indica qué porcentaje de varianza es explicado por el modelo
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def print_metrics(metrics, dataset_name="Dataset"):
    """
    Imprime métricas de forma legible.

    Args:
        metrics (dict): Diccionario con métricas
        dataset_name (str): Nombre del conjunto de datos (ej: "Train", "Test")
    """
    print(f"\n{'='*60}")
    print(f"Métricas en {dataset_name}:")
    print(f"{'='*60}")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")
    print(f"  MAE:  ${metrics['mae']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)")
    print(f"{'='*60}\n")


def train_model(model_type='random_forest', use_mlflow=True):
    """
    Entrena un modelo de regresión completo.

    Args:
        model_type (str): Tipo de modelo ('linear_regression' o 'random_forest')
        use_mlflow (bool): Si True, registra el experimento en MLFlow

    Returns:
        tuple: (pipeline, metrics_dict) - modelo entrenado y métricas

    Proceso:
        1. Carga y valida datos
        2. Separa train/test
        3. Crea pipeline de preprocesamiento + modelo
        4. Entrena el modelo
        5. Evalúa métricas
        6. Registra en MLFlow (opcional)
        7. Guarda el modelo
    """
    print(f"\n{'='*60}")
    print(f"Iniciando entrenamiento: {model_type}")
    print(f"{'='*60}\n")

    # 1. Cargar y validar datos
    df = load_data()
    validate_data(df)

    # 2. Preparar características
    X, y = prepare_features(df, include_target=True)
    print(f"Características: {X.shape}")
    print(f"Variable objetivo: {y.shape}")

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {X_train.shape[0]} muestras")
    print(f"Test set: {X_test.shape[0]} muestras")

    # 4. Crear preprocesador
    preprocessor = create_preprocessor()

    # 5. Seleccionar modelo
    if model_type == 'linear_regression':
        model = LinearRegression()
        model_params = {}
    elif model_type == 'random_forest':
        model = RandomForestRegressor(**RF_PARAMS)
        model_params = RF_PARAMS
    elif model_type == 'random_forest_deep':
        model = RandomForestRegressor(**RF_DEEP_PARAMS)
        model_params = RF_DEEP_PARAMS
    elif model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM no está instalado. Ejecuta: uv sync --extra regresion-automoviles")
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model_params = LGBM_PARAMS
    elif model_type == 'knn':
        model = KNeighborsRegressor(**KNN_PARAMS)
        model_params = KNN_PARAMS
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    # 6. Crear pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # 7. Entrenar modelo
    print(f"\nEntrenando modelo...")
    pipeline.fit(X_train, y_train)
    print("Entrenamiento completado")

    # 8. Hacer predicciones
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # 9. Calcular métricas
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print_metrics(train_metrics, "Entrenamiento")
    print_metrics(test_metrics, "Prueba")

    # 10. Registrar en MLFlow
    if use_mlflow:
        if not MLFLOW_AVAILABLE:
            print("ADVERTENCIA: MLFlow no está instalado. Saltando registro de experimentos.")
            print("Para instalar: uv sync --extra regresion-automoviles")
        else:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=model_type):
                # Registrar parámetros
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("test_size", TEST_SIZE)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("n_samples_train", X_train.shape[0])

                # Registrar parámetros específicos del modelo
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)

                # Registrar métricas de train
                mlflow.log_metric("train_rmse", train_metrics['rmse'])
                mlflow.log_metric("train_mae", train_metrics['mae'])
                mlflow.log_metric("train_r2", train_metrics['r2'])

                # Registrar métricas de test
                mlflow.log_metric("test_rmse", test_metrics['rmse'])
                mlflow.log_metric("test_mae", test_metrics['mae'])
                mlflow.log_metric("test_r2", test_metrics['r2'])

                # Guardar modelo en MLFlow
                mlflow.sklearn.log_model(pipeline, "model")

                print("Experimento registrado en MLFlow")

    # 11. Guardar modelo localmente
    save_model(pipeline, model_type, test_metrics['r2'])

    # 12. Retornar pipeline y métricas
    metrics_dict = {
        'train': train_metrics,
        'test': test_metrics,
        'model_type': model_type
    }

    return pipeline, metrics_dict


def save_model(pipeline, model_type, test_r2):
    """
    Guarda el modelo entrenado y sus metadatos.

    Args:
        pipeline (Pipeline): Pipeline de sklearn entrenado
        model_type (str): Tipo de modelo
        test_r2 (float): R² en conjunto de prueba

    Notas:
        - Usa joblib para guardar el pipeline (más eficiente que pickle)
        - Guarda metadatos en JSON para facilitar la carga posterior
    """
    # Crear directorio si no existe
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # Guardar pipeline
    model_path = MODELS_DIR / f"{model_type}_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModelo guardado en: {model_path}")

    # Guardar metadatos
    metadata = {
        'model_type': model_type,
        'test_r2': float(test_r2),
        'model_file': str(model_path.name)
    }

    metadata_path = MODELS_DIR / f"{model_type}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadatos guardados en: {metadata_path}")


def compare_models(use_mlflow=True):
    """
    Entrena y compara múltiples modelos.

    Args:
        use_mlflow (bool): Si True, registra experimentos en MLFlow

    Returns:
        pd.DataFrame: Tabla comparativa con métricas de todos los modelos

    Notas:
        Útil para experimentación y selección del mejor modelo.
    """
    models_to_compare = [
        'random_forest',
        'random_forest_deep',
        'lightgbm',
        'knn',
        'linear_regression'
    ]
    results = []

    for model_type in models_to_compare:
        print(f"\n{'#'*60}")
        print(f"Entrenando: {model_type}")
        print(f"{'#'*60}")

        try:
            pipeline, metrics = train_model(model_type, use_mlflow=use_mlflow)

            results.append({
                'Modelo': model_type,
                'RMSE (Test)': metrics['test']['rmse'],
                'MAE (Test)': metrics['test']['mae'],
                'R² (Test)': metrics['test']['r2'],
                'R² (Train)': metrics['train']['r2'],
                'Overfitting': metrics['train']['r2'] - metrics['test']['r2']
            })
        except Exception as e:
            print(f"ERROR al entrenar {model_type}: {e}")
            print("Saltando este modelo...\n")

    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('R² (Test)', ascending=False)

    print(f"\n{'='*80}")
    print("COMPARACIÓN DE MODELOS")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*80}\n")

    return comparison_df


if __name__ == "__main__":
    # Código de ejemplo para probar el módulo

    # Opción 1: Entrenar un solo modelo
    print("Opción 1: Entrenar Random Forest")
    pipeline, metrics = train_model('random_forest', use_mlflow=True)

    # Opción 2: Comparar múltiples modelos
    # print("Opción 2: Comparar modelos")
    # comparison_df = compare_models()

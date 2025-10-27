"""
Módulo de Predicción

Este módulo contiene funciones para cargar modelos entrenados
y realizar predicciones sobre datos nuevos.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from config import (
    MODELS_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def load_model(model_path=None):
    """
    Carga un modelo previamente entrenado.

    Args:
        model_path (str o Path, opcional): Ruta al archivo del modelo.
                                          Si es None, busca modelos en MODELS_DIR

    Returns:
        Pipeline: Modelo cargado listo para hacer predicciones

    Raises:
        FileNotFoundError: Si no se encuentra el modelo
    """
    if model_path is None:
        # Buscar el mejor modelo disponible
        model_path = find_best_model()

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    # Cargar modelo usando joblib
    model = joblib.load(model_path)
    print(f"Modelo cargado: {model_path.name}")

    return model


def find_best_model():
    """
    Encuentra el modelo con mejor desempeño en el directorio de modelos.

    Returns:
        Path: Ruta al mejor modelo

    Notas:
        Busca archivos JSON con metadatos y selecciona el de mayor R²
    """
    metadata_files = list(MODELS_DIR.glob("*_metadata.json"))

    if not metadata_files:
        raise FileNotFoundError("No se encontraron modelos entrenados")

    best_r2 = -1
    best_model_file = None

    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        r2 = metadata.get('test_r2', 0)
        if r2 > best_r2:
            best_r2 = r2
            best_model_file = metadata.get('model_file')

    if best_model_file is None:
        raise ValueError("No se pudo determinar el mejor modelo")

    best_model_path = MODELS_DIR / best_model_file
    print(f"Mejor modelo encontrado: {best_model_file} (R² = {best_r2:.4f})")

    return best_model_path


def validate_input(data):
    """
    Valida que los datos de entrada tengan las columnas correctas.

    Args:
        data (pd.DataFrame o dict): Datos de entrada

    Returns:
        pd.DataFrame: DataFrame validado

    Raises:
        ValueError: Si faltan columnas requeridas
    """
    # Convertir dict a DataFrame si es necesario
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Verificar que sea un DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Los datos deben ser un DataFrame o diccionario")

    # Verificar columnas requeridas
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    # Seleccionar solo las columnas necesarias (en el orden correcto)
    data_validated = data[required_columns].copy()

    return data_validated


def predict_single(model, input_data):
    """
    Realiza predicción para una sola instancia.

    Args:
        model (Pipeline): Modelo cargado
        input_data (dict): Diccionario con las características del automóvil

    Returns:
        float: Precio predicho

    Ejemplo:
        >>> input_data = {
        ...     'año': 2020,
        ...     'kilometraje': 50000,
        ...     'marca': 'Toyota',
        ...     'tipo_combustible': 'Gasolina',
        ...     # ... resto de características
        ... }
        >>> precio = predict_single(model, input_data)
    """
    # Validar entrada
    df = validate_input(input_data)

    # Hacer predicción
    prediction = model.predict(df)[0]

    return prediction


def predict_batch(model, input_data):
    """
    Realiza predicciones para múltiples instancias.

    Args:
        model (Pipeline): Modelo cargado
        input_data (pd.DataFrame): DataFrame con múltiples automóviles

    Returns:
        np.ndarray: Array con los precios predichos

    Notas:
        Útil para procesar múltiples predicciones de forma eficiente.
    """
    # Validar entrada
    df = validate_input(input_data)

    # Hacer predicciones
    predictions = model.predict(df)

    return predictions


def predict_with_confidence(model, input_data, n_estimators=None):
    """
    Realiza predicción con intervalo de confianza (solo para Random Forest).

    Args:
        model (Pipeline): Modelo Random Forest cargado
        input_data (dict o DataFrame): Datos de entrada
        n_estimators (int, opcional): Número de estimadores a usar

    Returns:
        dict: Diccionario con predicción, intervalo de confianza y desviación

    Notas:
        Random Forest permite estimar incertidumbre usando las predicciones
        de cada árbol individual del bosque.
    """
    # Validar entrada
    df = validate_input(input_data)

    # Obtener el modelo Random Forest del pipeline
    rf_model = model.named_steps.get('regressor')

    if not hasattr(rf_model, 'estimators_'):
        # No es un Random Forest o no está entrenado
        prediction = model.predict(df)[0]
        return {
            'prediction': prediction,
            'confidence_interval': None,
            'std': None
        }

    # Transformar datos con el preprocesador
    X_processed = model.named_steps['preprocessor'].transform(df)

    # Obtener predicciones de cada árbol individual
    # Esto permite calcular incertidumbre
    tree_predictions = np.array([
        tree.predict(X_processed)[0]
        for tree in rf_model.estimators_
    ])

    # Calcular estadísticas
    prediction = tree_predictions.mean()
    std = tree_predictions.std()
    confidence_interval = (
        prediction - 1.96 * std,  # Límite inferior (95% confianza)
        prediction + 1.96 * std   # Límite superior (95% confianza)
    )

    return {
        'prediction': prediction,
        'confidence_interval': confidence_interval,
        'std': std,
        'min': tree_predictions.min(),
        'max': tree_predictions.max()
    }


def create_example_input():
    """
    Crea un ejemplo de datos de entrada para testing.

    Returns:
        dict: Diccionario con valores de ejemplo

    Notas:
        Útil para probar el módulo y como referencia de formato de entrada.
    """
    example = {
        'marca': 'Toyota',
        'tipo_carroceria': 'SUV',
        'año': 2020,
        'kilometraje': 50000,
        'tipo_combustible': 'Gasolina',
        'transmision': 'Automática',
        'cilindrada': 2000,
        'potencia': 150,
        'peso': 1500,
        'consumo': 8.5,
        'color': 'Blanco',
        'edad_propietarios': 1,
        'calificacion_estado': 8.5,
        'region_venta': 'Centro'
    }
    return example


def format_prediction(prediction, with_currency=True):
    """
    Formatea la predicción para mostrarla de forma legible.

    Args:
        prediction (float): Valor de la predicción
        with_currency (bool): Si True, incluye símbolo de moneda

    Returns:
        str: Predicción formateada
    """
    if with_currency:
        return f"${prediction:,.2f} USD"
    else:
        return f"{prediction:,.2f}"


if __name__ == "__main__":
    # Código de ejemplo para probar el módulo
    print("Probando módulo de predicción...\n")

    # Cargar modelo
    try:
        model = load_model()
    except FileNotFoundError:
        print("ERROR: No se encontró un modelo entrenado.")
        print("Ejecuta primero el módulo training.py para entrenar un modelo.")
        exit(1)

    # Crear ejemplo de entrada
    input_data = create_example_input()
    print("\nDatos de entrada:")
    for key, value in input_data.items():
        print(f"  {key}: {value}")

    # Hacer predicción simple
    print("\n" + "="*60)
    print("PREDICCIÓN SIMPLE")
    print("="*60)
    precio = predict_single(model, input_data)
    print(f"Precio predicho: {format_prediction(precio)}")

    # Hacer predicción con confianza
    print("\n" + "="*60)
    print("PREDICCIÓN CON INTERVALO DE CONFIANZA")
    print("="*60)
    result = predict_with_confidence(model, input_data)
    print(f"Predicción: {format_prediction(result['prediction'])}")

    if result['confidence_interval'] is not None:
        lower, upper = result['confidence_interval']
        print(f"Intervalo de confianza (95%):")
        print(f"  Mínimo: {format_prediction(lower)}")
        print(f"  Máximo: {format_prediction(upper)}")
        print(f"Desviación estándar: ${result['std']:,.2f}")

    # Ejemplo de predicción por lotes
    print("\n" + "="*60)
    print("PREDICCIÓN POR LOTES")
    print("="*60)

    # Crear varios ejemplos
    batch_data = pd.DataFrame([
        create_example_input(),
        {**create_example_input(), 'año': 2015, 'kilometraje': 120000},
        {**create_example_input(), 'marca': 'BMW', 'tipo_combustible': 'Híbrido'}
    ])

    predictions = predict_batch(model, batch_data)
    print(f"Predicciones para {len(predictions)} automóviles:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {format_prediction(pred)}")

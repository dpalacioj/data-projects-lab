"""
Módulo de Preprocesamiento

Este módulo contiene funciones para cargar, limpiar y transformar datos
antes de entrenar o usar modelos de machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

from config import (
    RAW_DATA_FILE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE
)


def load_data(file_path=None):
    """
    Carga el dataset desde un archivo parquet o CSV.

    Args:
        file_path (str o Path, opcional): Ruta al archivo de datos.
                                         Si es None, usa la ruta por defecto.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados

    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    if file_path is None:
        file_path = RAW_DATA_FILE

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    # Cargar según la extensión del archivo
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Formato no soportado: {file_path.suffix}")

    print(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    return df


def validate_data(df):
    """
    Valida que el DataFrame tenga las columnas esperadas y no tenga valores nulos.

    Args:
        df (pd.DataFrame): DataFrame a validar

    Returns:
        bool: True si la validación es exitosa

    Raises:
        ValueError: Si faltan columnas o hay valores nulos
    """
    # Verificar que existan todas las columnas requeridas
    expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Faltan columnas en el dataset: {missing_columns}")

    # Verificar valores nulos
    null_counts = df[expected_columns].isnull().sum()
    if null_counts.any():
        print("Advertencia: Se encontraron valores nulos:")
        print(null_counts[null_counts > 0])
        # En producción, podríamos decidir eliminar o imputar estos valores

    return True


def create_preprocessor():
    """
    Crea un pipeline de preprocesamiento que:
    - Escala variables numéricas (StandardScaler)
    - Codifica variables categóricas (OneHotEncoder)

    Returns:
        ColumnTransformer: Transformador configurado para el preprocesamiento

    Notas:
        - StandardScaler: Normaliza features numéricas a media=0 y std=1
          Esto es importante para modelos sensibles a la escala como regresión lineal.

        - OneHotEncoder: Convierte categorías en variables binarias (0/1).
          handle_unknown='ignore' permite manejar categorías no vistas en train.
    """
    # Transformador para variables numéricas
    numeric_transformer = StandardScaler()

    # Transformador para variables categóricas
    # handle_unknown='ignore': Si aparece una categoría nueva, no genera error
    # sparse_output=False: Retorna array denso en lugar de matriz sparse
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    # Combinar ambos transformadores
    # Esto permite aplicar diferentes transformaciones a diferentes columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Eliminar columnas no especificadas
    )

    return preprocessor


def prepare_features(df, include_target=True):
    """
    Separa características (X) de la variable objetivo (y).

    Args:
        df (pd.DataFrame): DataFrame con todos los datos
        include_target (bool): Si True, retorna también la variable objetivo

    Returns:
        tuple: (X, y) si include_target=True, solo X si include_target=False

    Notas:
        Esta función es útil tanto para entrenamiento (necesitamos X e y)
        como para predicción (solo necesitamos X).
    """
    # Seleccionar solo las columnas de características
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns].copy()

    if include_target:
        if TARGET_VARIABLE not in df.columns:
            raise ValueError(f"Variable objetivo '{TARGET_VARIABLE}' no encontrada")
        y = df[TARGET_VARIABLE].copy()
        return X, y

    return X


def get_feature_names(preprocessor):
    """
    Obtiene los nombres de las características después del preprocesamiento.

    Args:
        preprocessor (ColumnTransformer): Preprocesador ya ajustado (fitted)

    Returns:
        list: Lista con nombres de todas las características

    Notas:
        Útil para interpretar importancia de variables en modelos como Random Forest.
        Las variables categóricas se expanden en múltiples columnas con OneHotEncoder.
    """
    # Nombres de features numéricas (permanecen igual)
    feature_names = list(NUMERIC_FEATURES)

    # Nombres de features categóricas (se expanden con OneHotEncoder)
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_feature_names)

    return feature_names


def clean_outliers(df, columns=None, n_std=3):
    """
    Remueve outliers extremos basándose en desviación estándar.

    Args:
        df (pd.DataFrame): DataFrame a limpiar
        columns (list, opcional): Columnas a analizar. Si None, usa todas las numéricas.
        n_std (int): Número de desviaciones estándar para considerar outlier

    Returns:
        pd.DataFrame: DataFrame sin outliers extremos

    Notas:
        Outliers son valores que están más allá de n_std desviaciones estándar
        de la media. Pueden afectar negativamente el rendimiento del modelo.
    """
    if columns is None:
        columns = NUMERIC_FEATURES

    df_clean = df.copy()
    initial_count = len(df_clean)

    for col in columns:
        if col in df_clean.columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            # Mantener solo valores dentro de n_std desviaciones estándar
            df_clean = df_clean[
                (df_clean[col] >= mean - n_std * std) &
                (df_clean[col] <= mean + n_std * std)
            ]

    removed_count = initial_count - len(df_clean)
    if removed_count > 0:
        print(f"Outliers removidos: {removed_count} ({removed_count/initial_count*100:.2f}%)")

    return df_clean


if __name__ == "__main__":
    # Código de ejemplo para probar el módulo
    print("Probando módulo de preprocesamiento...")

    # Cargar datos
    df = load_data()
    print(f"\nDatos cargados: {df.shape}")

    # Validar datos
    validate_data(df)
    print("Validación exitosa")

    # Preparar características
    X, y = prepare_features(df)
    print(f"\nCaracterísticas (X): {X.shape}")
    print(f"Variable objetivo (y): {y.shape}")

    # Crear preprocesador
    preprocessor = create_preprocessor()
    print("\nPreprocesador creado exitosamente")

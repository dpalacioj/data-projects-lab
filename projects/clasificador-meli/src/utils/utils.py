"""
Utilidades para manejo de configuraciones y predicciones del modelo.
"""
import json


def load_model_config(path):
    """
    Carga configuración de hiperparámetros desde un archivo JSON.

    Args:
        path: Ruta al archivo JSON con hiperparámetros

    Returns:
        Diccionario con la configuración del modelo
    """
    with open(path) as f:
        return json.load(f)


def save_model_config(config, path):
    """
    Guarda configuración de hiperparámetros en un archivo JSON.

    Args:
        config: Diccionario con hiperparámetros
        path: Ruta donde guardar el archivo JSON
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
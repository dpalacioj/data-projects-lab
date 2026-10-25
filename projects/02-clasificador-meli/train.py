# train.py
"""
Script de entrenamiento del modelo clasificador Meli.

Uso:
    python train.py              # Entrenar con hiperparámetros base
    python train.py --optimize   # Entrenar con optimización de hiperparámetros
"""
from src.models.train_model import main

if __name__ == "__main__":
    main(optimize=False)

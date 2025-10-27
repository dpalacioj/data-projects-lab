#!/usr/bin/env python3
"""
Script rápido de entrenamiento para testing
Entrena un modelo Random Forest simple
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from training import train_model

print("Iniciando entrenamiento de modelo...")
print("="*60)

# Entrenar modelo Random Forest sin MLFlow para ir más rápido
pipeline, metrics = train_model('random_forest', use_mlflow=False)

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"Modelo guardado exitosamente")
print(f"R² en test: {metrics['test']['r2']:.4f}")
print(f"RMSE en test: ${metrics['test']['rmse']:,.2f}")
print(f"MAE en test: ${metrics['test']['mae']:,.2f}")

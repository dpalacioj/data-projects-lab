#!/usr/bin/env python3
"""
Script para entrenar y comparar todos los modelos disponibles

Este script:
1. Entrena múltiples modelos de regresión
2. Compara sus métricas de desempeño
3. Identifica el mejor modelo basado en R²
4. Guarda todos los modelos entrenados

Uso:
    python train_all_models.py

O con MLFlow:
    python train_all_models.py --mlflow
"""

import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.append('src')

from training import compare_models, train_model


def main():
    """
    Función principal para entrenar y comparar modelos
    """
    parser = argparse.ArgumentParser(description='Entrenar y comparar todos los modelos')
    parser.add_argument(
        '--mlflow',
        action='store_true',
        help='Registrar experimentos en MLFlow'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['random_forest', 'random_forest_deep', 'lightgbm', 'knn', 'linear_regression'],
        help='Modelos específicos a entrenar (por defecto: todos)'
    )

    args = parser.parse_args()

    print("="*80)
    print("ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
    print("="*80)
    print(f"MLFlow: {'Activado' if args.mlflow else 'Desactivado'}")
    print("="*80)

    if args.models:
        # Entrenar solo modelos especificados
        print(f"\nEntrenando modelos especificados: {', '.join(args.models)}")
        results = []

        for model_type in args.models:
            print(f"\n{'#'*60}")
            print(f"Entrenando: {model_type}")
            print(f"{'#'*60}")

            try:
                pipeline, metrics = train_model(model_type, use_mlflow=args.mlflow)
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

        # Mostrar resultados
        if results:
            import pandas as pd
            comparison_df = pd.DataFrame(results)
            comparison_df = comparison_df.sort_values('R² (Test)', ascending=False)

            print(f"\n{'='*80}")
            print("COMPARACIÓN DE MODELOS")
            print(f"{'='*80}")
            print(comparison_df.to_string(index=False))
            print(f"{'='*80}\n")

            # Identificar mejor modelo
            best_model = comparison_df.iloc[0]
            print(f"\nMejor modelo: {best_model['Modelo']}")
            print(f"R² (Test): {best_model['R² (Test)']:.4f}")
            print(f"RMSE (Test): ${best_model['RMSE (Test)']:,.2f}")
    else:
        # Entrenar todos los modelos usando compare_models()
        print("\nEntrenando todos los modelos disponibles...\n")
        comparison_df = compare_models(use_mlflow=args.mlflow)

        # Identificar mejor modelo
        best_model = comparison_df.iloc[0]
        print(f"\nMejor modelo: {best_model['Modelo']}")
        print(f"R² (Test): {best_model['R² (Test)']:.4f}")
        print(f"RMSE (Test): ${best_model['RMSE (Test)']:,.2f}")

    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print("\nPróximos pasos:")
    print("  1. Revisar modelos guardados en: models/")
    print("  2. Usar el mejor modelo en la aplicación Gradio/FastAPI")
    print("  3. Si usaste MLFlow, explorar experimentos: mlflow ui")
    print()


if __name__ == "__main__":
    main()

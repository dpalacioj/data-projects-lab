#!/usr/bin/env python3
"""
Script de verificación rápida del proyecto
Verifica que todo esté configurado correctamente
"""

import sys
from pathlib import Path

def test_imports():
    """Verificar que todas las dependencias estén instaladas"""
    print("Verificando imports...")
    print("="*60)

    imports_ok = True

    # Dependencias básicas
    try:
        import pandas
        import numpy
        import sklearn
        print("  pandas, numpy, sklearn: OK")
    except ImportError as e:
        print(f"  pandas, numpy, sklearn: ERROR - {e}")
        imports_ok = False

    # Visualización
    try:
        import plotly
        print("  plotly: OK")
    except ImportError as e:
        print(f"  plotly: ERROR - {e}")
        imports_ok = False

    # MLFlow (opcional)
    try:
        import mlflow
        print("  mlflow: OK")
    except ImportError:
        print("  mlflow: NO INSTALADO (opcional, no afecta funcionamiento básico)")

    # Gradio
    try:
        import gradio
        print("  gradio: OK")
    except ImportError as e:
        print(f"  gradio: ERROR - {e}")
        imports_ok = False

    # FastAPI
    try:
        import fastapi
        import uvicorn
        print("  fastapi, uvicorn: OK")
    except ImportError as e:
        print(f"  fastapi, uvicorn: ERROR - {e}")
        imports_ok = False

    print("="*60)
    return imports_ok


def test_data():
    """Verificar que los datos existan"""
    print("\nVerificando datos...")
    print("="*60)

    data_csv = Path('data/raw/automoviles_usados.csv')
    data_parquet = Path('data/raw/automoviles_usados.parquet')

    data_ok = True

    if data_csv.exists():
        size_mb = data_csv.stat().st_size / (1024 * 1024)
        print(f"  CSV: OK ({size_mb:.2f} MB)")
    else:
        print("  CSV: NO ENCONTRADO")
        data_ok = False

    if data_parquet.exists():
        size_mb = data_parquet.stat().st_size / (1024 * 1024)
        print(f"  Parquet: OK ({size_mb:.2f} MB)")

        # Verificar número de registros
        try:
            import pandas as pd
            df = pd.read_parquet(data_parquet)
            print(f"  Registros: {len(df):,}")
            print(f"  Columnas: {len(df.columns)}")
        except Exception as e:
            print(f"  Error al leer datos: {e}")
            data_ok = False
    else:
        print("  Parquet: NO ENCONTRADO")
        data_ok = False

    if not data_ok:
        print("\n  ADVERTENCIA: Los datos deben estar incluidos con el proyecto.")
        print("  Contacta al instructor si no encuentras los datos.")

    print("="*60)
    return data_ok


def test_model():
    """Verificar que el modelo exista"""
    print("\nVerificando modelo...")
    print("="*60)

    model_file = Path('models/random_forest_model.pkl')
    metadata_file = Path('models/random_forest_metadata.json')

    model_ok = True

    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  Modelo: OK ({size_mb:.2f} MB)")
    else:
        print("  Modelo: NO ENCONTRADO")
        model_ok = False

    if metadata_file.exists():
        print(f"  Metadata: OK")

        # Leer metadata
        try:
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            print(f"  Tipo de modelo: {metadata.get('model_type')}")
            print(f"  R² (test): {metadata.get('test_r2', 0):.4f}")
        except Exception as e:
            print(f"  Error al leer metadata: {e}")
    else:
        print("  Metadata: NO ENCONTRADO")
        model_ok = False

    if not model_ok:
        print("\n  Ejecuta: uv run python train_quick.py")

    print("="*60)
    return model_ok


def test_modules():
    """Verificar que los módulos del proyecto funcionen"""
    print("\nVerificando módulos del proyecto...")
    print("="*60)

    sys.path.append('src')

    modules_ok = True

    # Config
    try:
        from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        print(f"  config: OK ({len(NUMERIC_FEATURES)} numéricas, {len(CATEGORICAL_FEATURES)} categóricas)")
    except Exception as e:
        print(f"  config: ERROR - {e}")
        modules_ok = False

    # Preprocessing
    try:
        from preprocessing import load_data, create_preprocessor
        print("  preprocessing: OK")
    except Exception as e:
        print(f"  preprocessing: ERROR - {e}")
        modules_ok = False

    # Training
    try:
        from training import calculate_metrics
        print("  training: OK")
    except Exception as e:
        print(f"  training: ERROR - {e}")
        modules_ok = False

    # Predict
    try:
        from predict import load_model, create_example_input
        print("  predict: OK")
    except Exception as e:
        print(f"  predict: ERROR - {e}")
        modules_ok = False

    print("="*60)
    return modules_ok


def test_prediction():
    """Verificar que se pueda hacer una predicción"""
    print("\nProbando predicción...")
    print("="*60)

    try:
        sys.path.append('src')
        from predict import load_model, predict_single, create_example_input, format_prediction

        model = load_model()
        input_data = create_example_input()
        precio = predict_single(model, input_data)

        print(f"  Modelo cargado: OK")
        print(f"  Predicción de prueba: {format_prediction(precio)}")
        print("  Sistema de predicción: OK")

        print("="*60)
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        print("="*60)
        return False


def main():
    """Ejecutar todas las verificaciones"""
    print("\n" + "="*60)
    print("VERIFICACIÓN DEL PROYECTO")
    print("="*60)

    # Verificar directorio
    cwd = Path.cwd()
    print(f"\nDirectorio actual: {cwd}")

    if not cwd.name == 'prediccion-precios-automoviles':
        print("\nADVERTENCIA: No estás en el directorio del proyecto")
        print("Deberías estar en: .../data-projects-lab/projects/prediccion-precios-automoviles")
        print("\n")

    # Ejecutar tests
    results = {
        'Imports': test_imports(),
        'Datos': test_data(),
        'Modelo': test_model(),
        'Módulos': test_modules(),
        'Predicción': test_prediction() if test_model() else False
    }

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)

    for test_name, result in results.items():
        status = "OK" if result else "ERROR"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")

    all_ok = all(results.values())

    print("="*60)

    if all_ok:
        print("\n¡TODO FUNCIONA CORRECTAMENTE!")
        print("\nPróximos pasos:")
        print("  1. uv run python app_gradio.py    (interfaz web)")
        print("  2. uv run uvicorn app_fastapi:app --reload  (API)")
        print("  3. Explorar notebooks en notebooks/")
    else:
        print("\nHay problemas que resolver:")
        if not results['Imports']:
            print("  - Instala dependencias: cd ../.. && uv sync --extra regresion-automoviles")
        if not results['Datos']:
            print("  - ADVERTENCIA: Los datos deben estar incluidos.")
            print("    Contacta al instructor si no los encuentras.")
        if not results['Modelo']:
            print("  - Entrena modelo: uv run python train_quick.py")

        print("\nConsulta TROUBLESHOOTING.md para más ayuda")

    print("\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""
Script de prueba para verificar el sistema de logging.

Carga datos de prueba y ejecuta el preprocesamiento para verificar
que los logs se generan correctamente.
"""
import json
import pandas as pd
from pathlib import Path
from src.features.preprocessing import Preprocessing
from src.utils.logger import setup_logger

# Configurar logger para este script
logger = setup_logger(__name__)

def main():
    """Función principal para probar el logging."""
    logger.info("=" * 60)
    logger.info("Iniciando prueba del sistema de logging")
    logger.info("=" * 60)

    # Cargar datos de prueba
    test_file = Path(__file__).parent / 'test_data' / 'single_product_example.json'

    logger.info(f"Cargando archivo de prueba: {test_file.name}")

    try:
        with open(test_file, 'r') as f:
            data = json.load(f)

        logger.info("✓ Archivo cargado exitosamente")

        # Convertir a DataFrame
        df = pd.DataFrame([data])
        logger.info(f"DataFrame creado | Shape: {df.shape} | Columnas: {len(df.columns)}")

        # Preprocesar datos
        logger.info("Iniciando preprocesamiento...")
        preprocessor = Preprocessing()
        df_processed = preprocessor.transform(df)

        logger.info(f"✓ Preprocesamiento completado | Shape final: {df_processed.shape}")

        # Verificar columnas
        logger.info(f"Columnas en datos procesados: {list(df_processed.columns)}")

        # Mostrar algunos valores
        logger.info("Valores de ejemplo del producto procesado:")
        for col in df_processed.columns[:5]:
            logger.info(f"  - {col}: {df_processed[col].iloc[0]}")

        logger.info("=" * 60)
        logger.info("✅ PRUEBA EXITOSA - Logging funcionando correctamente")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error durante la prueba: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

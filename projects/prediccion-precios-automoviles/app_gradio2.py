"""
Aplicación Gradio para Comparación de Modelos de Predicción de Precios

Esta aplicación permite comparar el desempeño de diferentes modelos de ML
lado a lado, mostrando tanto las predicciones como las métricas de evaluación.

Para ejecutar:
    python app_gradio2.py
"""

import gradio as gr
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Agregar src/ al path para importar módulos
sys.path.append(str(Path(__file__).parent / "src"))

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE
from preprocessing import load_data, prepare_features
from predict import predict_single, format_prediction


# =============================================================================
# CARGA DE MODELOS Y CÁLCULO DE MÉTRICAS
# =============================================================================

def load_all_models():
    """
    Carga todos los modelos disponibles en el directorio de modelos.

    Returns:
        dict: Diccionario con nombre_modelo: (modelo, metadata)
    """
    models = {}
    metadata_files = list(MODELS_DIR.glob("*_metadata.json"))

    print("Cargando modelos disponibles...")

    for metadata_file in metadata_files:
        # Evitar el archivo model_metadata.json (es un duplicado)
        if metadata_file.name == "model_metadata.json":
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        model_file = metadata.get('model_file')
        if not model_file:
            continue

        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            continue

        # Cargar modelo
        try:
            model = joblib.load(model_path)
            model_name = metadata.get('model_type', model_file.replace('.pkl', ''))
            models[model_name] = {
                'model': model,
                'metadata': metadata,
                'file': model_file
            }
            print(f"  ✓ {model_name} cargado")
        except Exception as e:
            print(f"  ✗ Error cargando {model_file}: {e}")

    return models


def compute_all_metrics(models):
    """
    Calcula métricas de test para todos los modelos.

    Args:
        models (dict): Diccionario de modelos cargados

    Returns:
        dict: Diccionario con nombre_modelo: {r2, mae, rmse}
    """
    print("\nCalculando métricas en conjunto de prueba...")

    # Cargar datos y hacer split (mismo que en entrenamiento)
    df = load_data()
    X, y = prepare_features(df, include_target=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    metrics = {}

    for model_name, model_info in models.items():
        model = model_info['model']

        try:
            # Predecir en test set
            y_pred = model.predict(X_test)

            # Calcular métricas
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            metrics[model_name] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }

            print(f"  ✓ {model_name}: R²={r2:.4f}, MAE=${mae:,.0f}")

        except Exception as e:
            print(f"  ✗ Error calculando métricas para {model_name}: {e}")
            metrics[model_name] = {
                'r2': 0.0,
                'mae': 0.0,
                'rmse': 0.0
            }

    return metrics


# Cargar modelos y métricas al iniciar
print("="*60)
print("INICIALIZANDO APLICACIÓN DE COMPARACIÓN DE MODELOS")
print("="*60)

try:
    AVAILABLE_MODELS = load_all_models()
    MODEL_METRICS = compute_all_metrics(AVAILABLE_MODELS)
    print(f"\n✓ {len(AVAILABLE_MODELS)} modelos cargados exitosamente\n")
    print("="*60)
except Exception as e:
    print(f"\n✗ Error en inicialización: {e}")
    AVAILABLE_MODELS = {}
    MODEL_METRICS = {}


# =============================================================================
# FUNCIÓN DE COMPARACIÓN
# =============================================================================

def compare_models(
    model1_name, model2_name,
    marca, tipo_carroceria, año, kilometraje, tipo_combustible,
    transmision, cilindrada, potencia, peso, consumo,
    color, edad_propietarios, calificacion_estado, region_venta
):
    """
    Compara dos modelos realizando predicciones y mostrando métricas.

    Returns:
        str: Reporte comparativo formateado
    """
    if not AVAILABLE_MODELS:
        return "ERROR: No se han cargado modelos. Verifica que existan modelos entrenados."

    if model1_name == model2_name:
        return "⚠️ Por favor selecciona dos modelos diferentes para comparar."

    try:
        # Crear diccionario con los datos de entrada
        input_data = {
            'marca': marca,
            'tipo_carroceria': tipo_carroceria,
            'año': int(año),
            'kilometraje': int(kilometraje),
            'tipo_combustible': tipo_combustible,
            'transmision': transmision,
            'cilindrada': int(cilindrada),
            'potencia': int(potencia),
            'peso': int(peso),
            'consumo': float(consumo),
            'color': color,
            'edad_propietarios': int(edad_propietarios),
            'calificacion_estado': float(calificacion_estado),
            'region_venta': region_venta
        }

        # Obtener modelos
        model1 = AVAILABLE_MODELS[model1_name]['model']
        model2 = AVAILABLE_MODELS[model2_name]['model']

        # Realizar predicciones
        pred1 = predict_single(model1, input_data)
        pred2 = predict_single(model2, input_data)

        # Obtener métricas
        metrics1 = MODEL_METRICS[model1_name]
        metrics2 = MODEL_METRICS[model2_name]

        # Calcular diferencias
        diff_precio = pred1 - pred2
        diff_precio_pct = (diff_precio / pred2) * 100 if pred2 != 0 else 0

        # Determinar mejor modelo
        mejor_r2 = model1_name if metrics1['r2'] > metrics2['r2'] else model2_name
        mejor_mae = model1_name if metrics1['mae'] < metrics2['mae'] else model2_name
        mejor_rmse = model1_name if metrics1['rmse'] < metrics2['rmse'] else model2_name

        # Formatear resultado
        resultado = f"""
╔══════════════════════════════════════════════════════════════╗
║              COMPARACIÓN DE MODELOS DE PREDICCIÓN            ║
╚══════════════════════════════════════════════════════════════╝

📊 MODELO 1: {model1_name.upper()}
{'─'*60}
   Predicción:        {format_prediction(pred1)}

   Métricas en Test Set:
   • R² Score:         {metrics1['r2']:.4f} ({metrics1['r2']*100:.2f}% varianza explicada)
   • MAE:              ${metrics1['mae']:,.2f} (error promedio)
   • RMSE:             ${metrics1['rmse']:,.2f} (penaliza errores grandes)


📊 MODELO 2: {model2_name.upper()}
{'─'*60}
   Predicción:        {format_prediction(pred2)}

   Métricas en Test Set:
   • R² Score:         {metrics2['r2']:.4f} ({metrics2['r2']*100:.2f}% varianza explicada)
   • MAE:              ${metrics2['mae']:,.2f} (error promedio)
   • RMSE:             ${metrics2['rmse']:,.2f} (penaliza errores grandes)


🔍 ANÁLISIS COMPARATIVO
{'─'*60}
   Diferencia en Predicción:
   • Absoluta:         ${abs(diff_precio):,.2f}
   • Relativa:         {abs(diff_precio_pct):.2f}%
   • Modelo 1 predice: {"MÁS ALTO" if diff_precio > 0 else "MÁS BAJO" if diff_precio < 0 else "IGUAL"}

   Mejor Modelo por Métrica:
   • Mayor R²:         {mejor_r2} ⭐
   • Menor MAE:        {mejor_mae} ⭐
   • Menor RMSE:       {mejor_rmse} ⭐


💡 INTERPRETACIÓN
{'─'*60}
   • R² más alto = Explica mejor la variabilidad de precios
   • MAE más bajo = Menor error promedio en predicciones
   • RMSE más bajo = Menos errores grandes (más confiable)

   ⚠️ IMPORTANTE: Las métricas son en el conjunto de PRUEBA (20% de datos
   no vistos durante entrenamiento), lo que indica el desempeño real.

{'═'*60}
"""

        return resultado

    except Exception as e:
        return f"❌ Error al realizar comparación: {str(e)}"


# =============================================================================
# INTERFAZ GRADIO
# =============================================================================

# Definir valores y opciones (mismo que app_gradio.py)
MARCAS = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'Mazda',
          'Hyundai', 'Kia', 'BMW', 'Mercedes-Benz', 'Audi']
TIPOS_CARROCERIA = ['Sedán', 'SUV', 'Hatchback', 'Pickup', 'Coupé', 'Minivan']
TIPOS_COMBUSTIBLE = ['Gasolina', 'Diesel', 'Híbrido', 'Eléctrico']
TRANSMISIONES = ['Manual', 'Automática']
COLORES = ['Blanco', 'Negro', 'Gris', 'Plata', 'Rojo', 'Azul', 'Verde', 'Amarillo']
REGIONES = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']

# Obtener nombres de modelos disponibles
model_names = sorted(AVAILABLE_MODELS.keys()) if AVAILABLE_MODELS else ['No hay modelos']

# Valores por defecto para comparación educativa
# LightGBM (mejor) vs Linear Regression (baseline simple)
default_model1 = 'lightgbm' if 'lightgbm' in model_names else model_names[0]
default_model2 = 'linear_regression' if 'linear_regression' in model_names else model_names[-1]


# Crear interfaz
with gr.Blocks(title="Comparación de Modelos - Predicción de Precios") as app:

    gr.Markdown("# 🔬 Comparación Interactiva de Modelos ML")
    gr.Markdown(
        """
        Esta aplicación permite **comparar dos modelos** de Machine Learning lado a lado.
        Ingresa las características de un automóvil y observa cómo diferentes modelos
        predicen su precio, junto con sus métricas de desempeño en datos de prueba.
        """
    )

    # Sección de selección de modelos
    gr.Markdown("## Selecciona los Modelos a Comparar")

    with gr.Row():
        model1_dropdown = gr.Dropdown(
            choices=model_names,
            label="🎯 Modelo 1",
            value=default_model1,
            info="Primer modelo para comparar"
        )
        model2_dropdown = gr.Dropdown(
            choices=model_names,
            label="🎯 Modelo 2",
            value=default_model2,
            info="Segundo modelo para comparar"
        )

    gr.Markdown("---")
    gr.Markdown("## Características del Automóvil")

    # Formulario de entrada (mismo layout que app_gradio.py)
    with gr.Row():
        # Columna 1: Información general
        with gr.Column():
            gr.Markdown("### Información General")
            marca = gr.Dropdown(choices=MARCAS, label="Marca", value="Toyota")
            tipo_carroceria = gr.Dropdown(choices=TIPOS_CARROCERIA, label="Tipo de Carrocería", value="SUV")
            año = gr.Number(label="Año de Fabricación", value=2020, minimum=2000, maximum=2024)
            kilometraje = gr.Number(label="Kilometraje", value=50000, minimum=0)

        # Columna 2: Motor y características técnicas
        with gr.Column():
            gr.Markdown("### Características Técnicas")
            tipo_combustible = gr.Dropdown(choices=TIPOS_COMBUSTIBLE, label="Tipo de Combustible", value="Gasolina")
            transmision = gr.Dropdown(choices=TRANSMISIONES, label="Transmisión", value="Automática")
            cilindrada = gr.Number(label="Cilindrada (cc)", value=2000, minimum=1000, maximum=5000)
            potencia = gr.Number(label="Potencia (HP)", value=150, minimum=50, maximum=500)

    with gr.Row():
        # Columna 3: Características físicas
        with gr.Column():
            gr.Markdown("### Características Físicas")
            peso = gr.Number(label="Peso (kg)", value=1500, minimum=800, maximum=3000)
            consumo = gr.Number(label="Consumo (L/100km)", value=8.5, minimum=0, maximum=20)
            color = gr.Dropdown(choices=COLORES, label="Color", value="Blanco")

        # Columna 4: Estado y ubicación
        with gr.Column():
            gr.Markdown("### Estado y Ubicación")
            edad_propietarios = gr.Number(label="Número de Propietarios Previos", value=1, minimum=1, maximum=5)
            calificacion_estado = gr.Slider(label="Calificación del Estado (1-10)", minimum=1, maximum=10, value=8.5, step=0.5)
            region_venta = gr.Dropdown(choices=REGIONES, label="Región de Venta", value="Centro")

    # Botón de comparación
    compare_button = gr.Button("🔍 Comparar Modelos", variant="primary", size="lg")

    # Salida de resultados
    output = gr.Textbox(
        label="Resultados de la Comparación",
        lines=35,
        interactive=False,
        show_copy_button=True
    )

    # Conectar el botón con la función de comparación
    compare_button.click(
        fn=compare_models,
        inputs=[
            model1_dropdown, model2_dropdown,
            marca, tipo_carroceria, año, kilometraje, tipo_combustible,
            transmision, cilindrada, potencia, peso, consumo,
            color, edad_propietarios, calificacion_estado, region_venta
        ],
        outputs=output
    )

    # Información adicional
    gr.Markdown("---")
    gr.Markdown(
        """
        ### 📚 Guía de Uso

        1. **Selecciona dos modelos** diferentes en los dropdowns superiores
        2. **Ingresa características** del automóvil en el formulario
        3. **Haz clic en "Comparar Modelos"** para ver resultados

        ### 📊 Métricas Explicadas

        - **R² (Coeficiente de Determinación)**: Mide qué % de variabilidad es explicado por el modelo (0-1, mayor es mejor)
        - **MAE (Error Absoluto Medio)**: Promedio de errores en dólares (menor es mejor)
        - **RMSE (Raíz del Error Cuadrático Medio)**: Similar a MAE pero penaliza más los errores grandes (menor es mejor)

        ### 💡 Casos de Uso Educativos

        - Compara **LightGBM vs Linear Regression** para ver diferencia entre modelos complejos y simples
        - Compara **Random Forest vs Random Forest Deep** para entender impacto de hiperparámetros
        - Compara **modelos similares** (LightGBM vs Random Forest) para elegir el mejor

        ### ⚠️ Nota Importante

        Las métricas mostradas son del **conjunto de prueba** (20% de datos no vistos),
        lo que representa el desempeño real del modelo en producción.

        ---

        **Proyecto Educativo** | Para fines académicos | MIT License
        """
    )


if __name__ == "__main__":
    # Lanzar aplicación
    print("\n" + "="*60)
    print("Iniciando servidor Gradio...")
    print("="*60)

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Puerto diferente al app_gradio.py (7860)
        share=False
    )

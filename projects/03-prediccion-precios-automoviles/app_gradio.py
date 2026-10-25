"""
Aplicación Gradio para Predicción de Precios de Automóviles

Esta aplicación crea una interfaz web interactiva que permite a los usuarios
ingresar características de un automóvil y obtener una predicción de su precio.

Para ejecutar:
    python app_gradio.py
"""

import gradio as gr
import sys
from pathlib import Path

# Agregar src/ al path para importar módulos
sys.path.append(str(Path(__file__).parent / "src"))

from predict import load_model, predict_single, format_prediction


# Cargar modelo al iniciar la aplicación
# Esto se hace una sola vez para mejorar el rendimiento
print("Cargando modelo...")
try:
    model = load_model()
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    print("Asegúrate de haber entrenado un modelo primero ejecutando el notebook 03_training.ipynb")
    model = None


def predict_price(
    marca, tipo_carroceria, año, kilometraje, tipo_combustible,
    transmision, cilindrada, potencia, peso, consumo,
    color, edad_propietarios, calificacion_estado, region_venta
):
    """
    Función que realiza la predicción del precio.

    Args:
        Parámetros individuales para cada característica del automóvil

    Returns:
        str: Mensaje con la predicción formateada

    Notas:
        Esta función es llamada automáticamente por Gradio cuando el usuario
        hace clic en el botón de predicción.
    """
    if model is None:
        return "ERROR: Modelo no cargado. Entrena un modelo primero."

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

        # Realizar predicción
        precio_predicho = predict_single(model, input_data)

        # Formatear resultado
        resultado = f"Precio estimado: {format_prediction(precio_predicho)}"

        return resultado

    except Exception as e:
        return f"Error al realizar predicción: {str(e)}"


# Definir valores por defecto y opciones
MARCAS = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'Mazda',
          'Hyundai', 'Kia', 'BMW', 'Mercedes-Benz', 'Audi']

TIPOS_CARROCERIA = ['Sedán', 'SUV', 'Hatchback', 'Pickup', 'Coupé', 'Minivan']

TIPOS_COMBUSTIBLE = ['Gasolina', 'Diesel', 'Híbrido', 'Eléctrico']

TRANSMISIONES = ['Manual', 'Automática']

COLORES = ['Blanco', 'Negro', 'Gris', 'Plata', 'Rojo', 'Azul', 'Verde', 'Amarillo']

REGIONES = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']


# Crear interfaz Gradio
with gr.Blocks(title="Predicción de Precios de Automóviles") as app:

    gr.Markdown("# Predicción de Precios de Automóviles Usados")
    gr.Markdown(
        "Ingrese las características del automóvil para obtener una estimación de su precio."
    )

    with gr.Row():
        # Columna 1: Información general
        with gr.Column():
            gr.Markdown("### Información General")
            marca = gr.Dropdown(
                choices=MARCAS,
                label="Marca",
                value="Toyota"
            )
            tipo_carroceria = gr.Dropdown(
                choices=TIPOS_CARROCERIA,
                label="Tipo de Carrocería",
                value="SUV"
            )
            año = gr.Number(
                label="Año de Fabricación",
                value=2020,
                minimum=2000,
                maximum=2024
            )
            kilometraje = gr.Number(
                label="Kilometraje",
                value=50000,
                minimum=0
            )

        # Columna 2: Motor y características técnicas
        with gr.Column():
            gr.Markdown("### Características Técnicas")
            tipo_combustible = gr.Dropdown(
                choices=TIPOS_COMBUSTIBLE,
                label="Tipo de Combustible",
                value="Gasolina"
            )
            transmision = gr.Dropdown(
                choices=TRANSMISIONES,
                label="Transmisión",
                value="Automática"
            )
            cilindrada = gr.Number(
                label="Cilindrada (cc)",
                value=2000,
                minimum=1000,
                maximum=5000
            )
            potencia = gr.Number(
                label="Potencia (HP)",
                value=150,
                minimum=50,
                maximum=500
            )

    with gr.Row():
        # Columna 3: Características físicas
        with gr.Column():
            gr.Markdown("### Características Físicas")
            peso = gr.Number(
                label="Peso (kg)",
                value=1500,
                minimum=800,
                maximum=3000
            )
            consumo = gr.Number(
                label="Consumo (L/100km)",
                value=8.5,
                minimum=0,
                maximum=20
            )
            color = gr.Dropdown(
                choices=COLORES,
                label="Color",
                value="Blanco"
            )

        # Columna 4: Estado y ubicación
        with gr.Column():
            gr.Markdown("### Estado y Ubicación")
            edad_propietarios = gr.Number(
                label="Número de Propietarios Previos",
                value=1,
                minimum=1,
                maximum=5
            )
            calificacion_estado = gr.Slider(
                label="Calificación del Estado (1-10)",
                minimum=1,
                maximum=10,
                value=8.5,
                step=0.5
            )
            region_venta = gr.Dropdown(
                choices=REGIONES,
                label="Región de Venta",
                value="Centro"
            )

    # Botón de predicción
    predict_button = gr.Button("Predecir Precio", variant="primary")

    # Salida de resultados
    output = gr.Textbox(
        label="Resultado",
        lines=2,
        interactive=False
    )

    # Conectar el botón con la función de predicción
    predict_button.click(
        fn=predict_price,
        inputs=[
            marca, tipo_carroceria, año, kilometraje, tipo_combustible,
            transmision, cilindrada, potencia, peso, consumo,
            color, edad_propietarios, calificacion_estado, region_venta
        ],
        outputs=output
    )

    # Agregar información adicional al pie
    gr.Markdown("---")
    gr.Markdown(
        """
        ### Notas:
        - Los precios son estimaciones basadas en el modelo entrenado
        - La precisión depende de la calidad de los datos de entrenamiento
        - Este proyecto es con fines educativos
        """
    )


if __name__ == "__main__":
    # Lanzar aplicación
    app.launch(
        server_name="0.0.0.0",  # Permite acceso desde otras máquinas en la red
        server_port=7860,        # Puerto por defecto de Gradio
        share=False             
    )

"""
API FastAPI para Predicción de Precios de Automóviles

Esta API permite realizar predicciones de precios mediante requests HTTP.
Es ideal para integrar el modelo en otras aplicaciones o servicios.

Para ejecutar:
    uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000

Documentación interactiva disponible en:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Agregar src/ al path para importar módulos
sys.path.append(str(Path(__file__).parent / "src"))

from predict import load_model, predict_single, format_prediction


# Crear aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Precios de Automóviles",
    description="API REST para predecir precios de automóviles usados usando Machine Learning",
    version="1.0.0"
)


# Modelo Pydantic para validación de entrada
# Esto asegura que los datos recibidos tengan el formato correcto
class AutomovilInput(BaseModel):
    """
    Modelo de datos para entrada de predicción.

    Atributos:
        Características del automóvil necesarias para la predicción.

    Ejemplo:
        {
            "marca": "Toyota",
            "tipo_carroceria": "SUV",
            "año": 2020,
            "kilometraje": 50000,
            ...
        }
    """
    marca: str = Field(..., description="Marca del automóvil")
    tipo_carroceria: str = Field(..., description="Tipo de carrocería")
    año: int = Field(..., ge=2000, le=2024, description="Año de fabricación")
    kilometraje: int = Field(..., ge=0, description="Kilómetros recorridos")
    tipo_combustible: str = Field(..., description="Tipo de combustible")
    transmision: str = Field(..., description="Tipo de transmisión")
    cilindrada: int = Field(..., ge=1000, le=5000, description="Cilindrada en cc")
    potencia: int = Field(..., ge=50, le=500, description="Potencia en HP")
    peso: int = Field(..., ge=800, le=3000, description="Peso en kg")
    consumo: float = Field(..., ge=0, le=20, description="Consumo en L/100km")
    color: str = Field(..., description="Color del vehículo")
    edad_propietarios: int = Field(..., ge=1, le=5, description="Número de propietarios previos")
    calificacion_estado: float = Field(..., ge=1, le=10, description="Calificación del estado (1-10)")
    region_venta: str = Field(..., description="Región donde se vende")

    class Config:
        # Ejemplo para la documentación automática
        json_schema_extra = {
            "example": {
                "marca": "Toyota",
                "tipo_carroceria": "SUV",
                "año": 2020,
                "kilometraje": 50000,
                "tipo_combustible": "Gasolina",
                "transmision": "Automática",
                "cilindrada": 2000,
                "potencia": 150,
                "peso": 1500,
                "consumo": 8.5,
                "color": "Blanco",
                "edad_propietarios": 1,
                "calificacion_estado": 8.5,
                "region_venta": "Centro"
            }
        }


# Modelo Pydantic para respuesta de predicción
class PredictionResponse(BaseModel):
    """
    Modelo de datos para respuesta de predicción.

    Atributos:
        precio_predicho: Precio estimado en USD
        precio_formateado: Precio en formato legible
        modelo_usado: Tipo de modelo utilizado
        status: Estado de la predicción
    """
    precio_predicho: float = Field(..., description="Precio predicho en USD")
    precio_formateado: str = Field(..., description="Precio formateado con moneda")
    modelo_usado: str = Field(..., description="Tipo de modelo utilizado")
    status: str = Field(default="success", description="Estado de la operación")


# Modelo para respuesta de health check
class HealthResponse(BaseModel):
    """
    Modelo de datos para health check.

    Atributos:
        status: Estado del servicio
        model_loaded: Si el modelo está cargado
        message: Mensaje adicional
    """
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Indica si el modelo está cargado")
    message: str = Field(..., description="Mensaje adicional")


# Variable global para el modelo
# Se carga al iniciar la aplicación
model = None
model_type = "unknown"


@app.on_event("startup")
async def startup_event():
    """
    Evento ejecutado al iniciar la aplicación.
    Carga el modelo en memoria para mejorar el rendimiento.
    """
    global model, model_type

    print("Iniciando aplicación...")
    print("Cargando modelo...")

    try:
        model = load_model()
        # Intentar obtener el tipo de modelo del pipeline
        if hasattr(model, 'named_steps'):
            regressor = model.named_steps.get('regressor')
            model_type = type(regressor).__name__
        print(f"Modelo cargado exitosamente: {model_type}")
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        print("La aplicación continuará pero las predicciones fallarán")


@app.get("/", tags=["General"])
async def root():
    """
    Endpoint raíz de la API.
    Proporciona información básica sobre el servicio.
    """
    return {
        "message": "API de Predicción de Precios de Automóviles",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Endpoint de health check.
    Verifica que el servicio esté funcionando y el modelo esté cargado.

    Returns:
        HealthResponse: Estado del servicio

    Ejemplo de uso:
        curl http://localhost:8000/health
    """
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message="Modelo no cargado. Entrena un modelo primero."
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message=f"Servicio operativo. Modelo cargado: {model_type}"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(automovil: AutomovilInput):
    """
    Endpoint para realizar predicciones de precios.

    Args:
        automovil (AutomovilInput): Datos del automóvil en formato JSON

    Returns:
        PredictionResponse: Predicción del precio

    Raises:
        HTTPException: Si el modelo no está cargado o hay error en la predicción

    Ejemplo de uso con curl:
        curl -X POST "http://localhost:8000/predict" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "marca": "Toyota",
                   "tipo_carroceria": "SUV",
                   "año": 2020,
                   "kilometraje": 50000,
                   "tipo_combustible": "Gasolina",
                   "transmision": "Automática",
                   "cilindrada": 2000,
                   "potencia": 150,
                   "peso": 1500,
                   "consumo": 8.5,
                   "color": "Blanco",
                   "edad_propietarios": 1,
                   "calificacion_estado": 8.5,
                   "region_venta": "Centro"
                 }'

    Ejemplo de uso con Python:
        import requests

        data = {
            "marca": "Toyota",
            "tipo_carroceria": "SUV",
            "año": 2020,
            ...
        }

        response = requests.post("http://localhost:8000/predict", json=data)
        print(response.json())
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Entrena un modelo primero ejecutando el notebook 03_training.ipynb"
        )

    try:
        # Convertir entrada Pydantic a diccionario
        input_data = automovil.model_dump()

        # Realizar predicción
        precio_predicho = predict_single(model, input_data)

        # Formatear precio
        precio_formateado = format_prediction(precio_predicho)

        return PredictionResponse(
            precio_predicho=float(precio_predicho),
            precio_formateado=precio_formateado,
            modelo_usado=model_type,
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar predicción: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def model_info():
    """
    Obtiene información sobre el modelo cargado.

    Returns:
        dict: Información del modelo

    Ejemplo de uso:
        curl http://localhost:8000/model-info
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )

    # Obtener información del modelo
    info = {
        "model_type": model_type,
        "model_loaded": True
    }

    # Si es un pipeline, obtener información adicional
    if hasattr(model, 'named_steps'):
        info["pipeline_steps"] = list(model.named_steps.keys())

        # Si es Random Forest, obtener número de árboles
        regressor = model.named_steps.get('regressor')
        if hasattr(regressor, 'n_estimators'):
            info["n_estimators"] = regressor.n_estimators
        if hasattr(regressor, 'max_depth'):
            info["max_depth"] = regressor.max_depth

    return info


if __name__ == "__main__":
    import uvicorn

    # Ejecutar aplicación
    # reload=True permite hot-reloading durante desarrollo
    # host="0.0.0.0" permite conexiones desde cualquier IP
    # port=8000 es el puerto estándar para APIs
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

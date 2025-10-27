# Predicción de Precios de Automóviles Usados

## Documentación Rápida

- **[QUICKSTART.md](QUICKSTART.md)** - Comandos directos para empezar en 5 minutos
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Guía completa de solución de problemas
- **Script de verificación:** `uv run python test_setup.py` - Verifica que todo funcione

## Descripción del Proyecto

Este proyecto educativo implementa un sistema completo de Machine Learning para predecir precios de automóviles usados. El objetivo es demostrar el flujo de trabajo completo de un proyecto de Data Science, desde la generación de datos hasta el deployment de modelos en producción.

### Objetivo Pedagógico

El proyecto está diseñado para estudiantes de IA y Analytics, mostrando:
- Generación de datos sintéticos realistas
- Análisis Exploratorio de Datos (EDA)
- Entrenamiento y evaluación de modelos
- Seguimiento de experimentos con MLFlow
- Modularización de código
- Creación de interfaces de usuario con Gradio
- Desarrollo de APIs REST con FastAPI

## Estructura del Proyecto

### Cookiecutter Data Science

Este proyecto utiliza la estructura de **Cookiecutter Data Science**, una plantilla estandarizada para proyectos de Data Science que promueve:
- Organización consistente de archivos y carpetas
- Separación entre datos, código y modelos
- Reproducibilidad y colaboración
- Mejores prácticas de la industria

**Referencia oficial:** https://cookiecutter-data-science.drivendata.org

**¿Qué es Cookiecutter Data Science?**
Es un estándar de facto para organizar proyectos de ML/DS que facilita:
1. **Navegación intuitiva**: Cualquier científico de datos puede entender la estructura
2. **Escalabilidad**: Estructura que crece con el proyecto
3. **Reproducibilidad**: Separación clara entre datos crudos y procesados
4. **Colaboración**: Equipo puede trabajar sin confusión

El proyecto sigue la estructura Cookiecutter Data Science:

```
prediccion-precios-automoviles/
│
├── data/
│   ├── raw/                    # Datos originales generados
│   ├── processed/              # Datos procesados
│   └── interim/                # Datos intermedios
│
├── models/                     # Modelos entrenados guardados
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_generacion_datos.ipynb    # Generación de datos sintéticos
│   ├── 02_eda.ipynb                 # Análisis exploratorio
│   └── 03_training.ipynb            # Entrenamiento con MLFlow
│
├── src/                        # Código fuente modular
│   ├── __init__.py
│   ├── config.py               # Configuración y constantes
│   ├── preprocessing.py        # Funciones de preprocesamiento
│   ├── training.py             # Funciones de entrenamiento
│   └── predict.py              # Funciones de predicción
│
├── reports/                    # Reportes y análisis
│   └── figures/                # Gráficos generados
│
├── app_gradio.py              # Aplicación web interactiva
├── app_fastapi.py             # API REST
└── README.md                  # Este archivo
```

## Instalación y Configuración

### Requisitos Previos

- Python 3.11 o superior
- uv (gestor de paquetes)

### Paso 1: Instalar Dependencias

Desde la raíz del repositorio `data-projects-lab`:

```bash
# Instalar dependencias del proyecto de regresión
uv sync --extra regresion-automoviles
```

Esto instalará automáticamente:
- pandas, numpy, scikit-learn (para ML)
- plotly (para visualizaciones)
- mlflow (para tracking de experimentos)
- gradio (para interfaz web)
- fastapi, uvicorn (para API REST)
- jupyter (para notebooks)

### Paso 2: Activar Entorno Virtual

```bash
# El entorno virtual está en la raíz del repositorio
source .venv/bin/activate  # En macOS/Linux
```

## Uso del Proyecto

### Dataset de Automóviles Usados

El proyecto incluye un dataset de 10,000 automóviles usados con 15 características:

**Variables Numéricas (8):**
- año: Año de fabricación (2010-2024)
- kilometraje: Kilómetros recorridos
- cilindrada: Tamaño del motor en cc
- potencia: Potencia en HP
- peso: Peso del vehículo en kg
- consumo: Consumo en L/100km
- edad_propietarios: Número de propietarios previos
- calificacion_estado: Estado del vehículo (1-10)

**Variables Categóricas (6):**
- marca: Marca del automóvil (Toyota, Honda, BMW, etc.)
- tipo_carroceria: Sedán, SUV, Hatchback, Pickup, Coupé, Minivan
- tipo_combustible: Gasolina, Diesel, Híbrido, Eléctrico
- transmision: Manual o Automática
- color: Color del vehículo
- region_venta: Región donde se vende

**Variable objetivo:**
- precio: Precio de venta en USD

Los datos están disponibles en `data/raw/automoviles_usados.parquet`

### 1. Análisis Exploratorio de Datos (EDA)

**Notebook:** `notebooks/01_eda.ipynb`

Este notebook realiza un análisis exhaustivo usando Plotly para visualizaciones interactivas:

**Análisis incluidos:**
- Estadísticas descriptivas
- Matriz de correlación
- Distribuciones de variables (histogramas y boxplots)
- Detección de outliers
- Relación entre características y precio
- Análisis por variables categóricas
- Análisis multivariado

**Insights generados:**
- Variables más correlacionadas con el precio
- Patrones de depreciación
- Diferencias de precio por marca, combustible, transmisión
- Recomendaciones para el modelado

**Ejecutar:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Entrenamiento de Modelos con MLFlow

**Notebook:** `notebooks/02_training.ipynb`

Este notebook entrena y compara dos modelos de regresión:

**Modelos:**
1. **Regresión Lineal**: Modelo baseline simple
2. **Random Forest**: Modelo avanzado con mejor capacidad predictiva

**Pipeline de preprocesamiento:**
- StandardScaler para variables numéricas
- OneHotEncoder para variables categóricas

**Métricas de evaluación:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coeficiente de determinación)

**Características de MLFlow:**
- Registro automático de parámetros
- Tracking de métricas train/test
- Versionado de modelos
- Comparación de experimentos

**Ejecutar:**
```bash
jupyter notebook notebooks/02_training.ipynb
```

El mejor modelo se guarda en `models/`

**Visualizar experimentos en MLFlow UI:**
```bash
cd projects/prediccion-precios-automoviles
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

Abrir en navegador: http://localhost:5000

### 3. Módulos Python Reutilizables

Los módulos en `src/` permiten usar el código fuera de los notebooks:

#### config.py
Centraliza configuración del proyecto:
- Rutas de datos y modelos
- Definición de características
- Parámetros de entrenamiento
- Hiperparámetros de modelos

#### preprocessing.py
Funciones de preprocesamiento:
- `load_data()`: Cargar datasets
- `validate_data()`: Validar integridad de datos
- `create_preprocessor()`: Crear pipeline de transformación
- `prepare_features()`: Separar X e y
- `clean_outliers()`: Remover outliers

**Ejemplo de uso:**
```python
from src.preprocessing import load_data, prepare_features

df = load_data()
X, y = prepare_features(df)
```

#### training.py
Funciones de entrenamiento:
- `train_model()`: Entrenar modelo completo con MLFlow
- `calculate_metrics()`: Calcular RMSE, MAE, R²
- `compare_models()`: Comparar múltiples modelos
- `save_model()`: Guardar modelo y metadatos

**Ejemplo de uso:**
```python
from src.training import train_model

pipeline, metrics = train_model('random_forest', use_mlflow=True)
```

#### predict.py
Funciones de predicción:
- `load_model()`: Cargar modelo entrenado
- `predict_single()`: Predicción para un automóvil
- `predict_batch()`: Predicciones para múltiples automóviles
- `predict_with_confidence()`: Predicción con intervalo de confianza

**Ejemplo de uso:**
```python
from src.predict import load_model, predict_single

model = load_model()
input_data = {
    'marca': 'Toyota',
    'año': 2020,
    'kilometraje': 50000,
    # ... resto de características
}
precio = predict_single(model, input_data)
print(f"Precio estimado: ${precio:,.2f}")
```

### 4. Aplicación Web con Gradio

**Archivo:** `app_gradio.py`

Interfaz web interactiva para realizar predicciones de forma visual.

**Características:**
- Formulario con todos los campos necesarios
- Valores por defecto razonables
- Dropdowns para variables categóricas
- Sliders para variables continuas
- Predicción en tiempo real

**Ejecutar:**
```bash
cd projects/prediccion-precios-automoviles
python app_gradio.py
```

Abrir en navegador: http://localhost:7860

**Captura de pantalla (conceptual):**
- Panel izquierdo: Información general (marca, modelo, año, km)
- Panel derecho: Características técnicas (motor, potencia, consumo)
- Panel inferior: Estado y ubicación
- Botón de predicción
- Resultado: Precio estimado en USD

### 5. API REST con FastAPI

**Archivo:** `app_fastapi.py`

API REST para integrar el modelo en otras aplicaciones.

**Endpoints disponibles:**

#### GET /
Información básica de la API

#### GET /health
Health check del servicio
```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Servicio operativo. Modelo cargado: RandomForestRegressor"
}
```

#### POST /predict
Realizar predicción
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
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
```

**Respuesta:**
```json
{
  "precio_predicho": 18500.00,
  "precio_formateado": "$18,500.00 USD",
  "modelo_usado": "RandomForestRegressor",
  "status": "success"
}
```

#### GET /model-info
Información del modelo cargado

**Ejecutar API:**
```bash
cd projects/prediccion-precios-automoviles
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000
```

**Documentación interactiva:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Ejemplo con Python requests:**
```python
import requests

data = {
    "marca": "BMW",
    "tipo_carroceria": "Sedán",
    "año": 2022,
    "kilometraje": 20000,
    "tipo_combustible": "Híbrido",
    "transmision": "Automática",
    "cilindrada": 2500,
    "potencia": 250,
    "peso": 1700,
    "consumo": 6.5,
    "color": "Negro",
    "edad_propietarios": 1,
    "calificacion_estado": 9.5,
    "region_venta": "Norte"
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Precio predicho: {result['precio_formateado']}")
```

## Flujo de Trabajo Completo

### Paso 1: Explorar Datos
```bash
jupyter notebook notebooks/01_eda.ipynb
# Ejecutar todas las celdas y analizar visualizaciones
```

### Paso 2: Entrenar Modelo
```bash
jupyter notebook notebooks/02_training.ipynb
# Ejecutar todas las celdas
```

### Paso 3: Usar Modelo

**Opción A: Interfaz Gradio**
```bash
python app_gradio.py
# Abrir http://localhost:7860
```

**Opción B: API FastAPI**
```bash
uvicorn app_fastapi:app --reload
# Usar curl o requests para hacer predicciones
```

**Opción C: Código Python**
```python
from src.predict import load_model, predict_single

model = load_model()
precio = predict_single(model, input_data)
```

## Conceptos Clave Aprendidos

### 1. Análisis Exploratorio de Datos
- Estadísticas descriptivas
- Correlaciones y multicolinealidad
- Detección de outliers
- Visualizaciones interactivas con Plotly
- Análisis univariado y multivariado

### 2. Machine Learning
- Preprocesamiento de datos (escalado, codificación)
- Pipelines de scikit-learn
- Modelos de regresión (Linear, Random Forest)
- Métricas de evaluación (RMSE, MAE, R²)
- Overfitting vs generalización

### 3. MLOps
- Tracking de experimentos con MLFlow
- Versionado de modelos
- Reproducibilidad
- Gestión de modelos en producción

### 4. Ingeniería de Software
- Modularización de código
- Separación de concerns
- Documentación
- Manejo de configuración

### 5. Deployment
- Interfaces de usuario (Gradio)
- APIs REST (FastAPI)
- Validación de datos (Pydantic)
- Documentación automática (Swagger)

## Mejoras Futuras

Este proyecto es una base educativa que puede extenderse:

### Datos
- Integrar datos reales de mercados de automóviles
- Agregar más características (equipamiento, historial de mantenimiento)
- Implementar data augmentation

### Modelos
- Probar modelos avanzados (XGBoost, LightGBM, CatBoost)
- Implementar stacking/ensemble de modelos
- Optimización de hiperparámetros con Optuna
- Validación cruzada

### Preprocesamiento
- Feature engineering (crear variables derivadas)
- Target encoding para categóricas
- Manejo de valores faltantes
- Detección automática de outliers

### Deployment
- Containerización con Docker
- CI/CD con GitHub Actions
- Monitoreo de predicciones
- A/B testing de modelos
- Logging estructurado

### Interfaz
- Dashboard con Streamlit o Plotly Dash
- Explicabilidad de predicciones (SHAP, LIME)
- Comparación de múltiples automóviles
- Histórico de predicciones

## Recursos Adicionales

### Documentación
- [Scikit-learn](https://scikit-learn.org/)
- [MLFlow](https://mlflow.org/)
- [Gradio](https://gradio.app/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Plotly](https://plotly.com/python/)

### Tutoriales
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

## Solución de Problemas

### Error: Modelo no encontrado
**Causa:** No se ha entrenado ningún modelo
**Solución:** Ejecutar el notebook `02_training.ipynb` o el script `uv run python train_quick.py`

### Error: Archivo de datos no encontrado
**Causa:** Los datos no están en la ubicación esperada
**Solución:** Verificar que existe `data/raw/automoviles_usados.parquet` - contactar al instructor si no está disponible

### Error: ModuleNotFoundError
**Causa:** Dependencias no instaladas
**Solución:** Desde la raíz del repositorio, ejecutar `uv sync --extra regresion-automoviles`

### Error: Puerto ocupado (Gradio/FastAPI)
**Causa:** Otra aplicación está usando el puerto
**Solución:** Cambiar el puerto o detener la otra aplicación

```python
# Para Gradio
app.launch(server_port=7861)  # Cambiar puerto

# Para FastAPI
uvicorn app_fastapi:app --port 8001  # Cambiar puerto
```

## Licencia

MIT License - Este proyecto es de código abierto y con fines educativos.

## Autor

David Palacio Jiménez - Proyecto educativo para estudiantes de IA y Analytics

## Contribuciones

Este es un proyecto educativo. Las contribuciones son bienvenidas para:
- Mejorar la documentación
- Agregar nuevos modelos
- Optimizar el código
- Corregir errores

## Contacto

Para preguntas o sugerencias sobre este proyecto educativo, por favor abre un issue en el repositorio.

# Ejercicio de Clasificación de Productos - MercadoLibre

## Contexto del Problema

MercadoLibre, siendo uno de los principales marketplaces de América Latina, necesita implementar un sistema automatizado para clasificar los productos listados en su plataforma como "nuevos" o "usados". Esta clasificación es crucial para:

- Mejorar la experiencia de búsqueda de los usuarios
- Garantizar la calidad y confiabilidad de las publicaciones
- Optimizar los algoritmos de recomendación
- Facilitar las decisiones de compra de los usuarios

## Objetivo Principal

Desarrollar una solución de Machine Learning que pueda predecir automáticamente si un artículo publicado en el marketplace de MercadoLibre es **"nuevo"** o **"usado"** basándose en las características disponibles del producto.

## Objetivos Específicos y Medibles

### 1. Precisión Mínima Requerida
- **Objetivo:** Alcanzar una precisión (accuracy) mínima del **86%** en el conjunto de datos de prueba
- **Métrica principal:** Accuracy
- **Umbral mínimo:** 0.75

### 2. Análisis de Métricas Complementarias
- **Objetivo:** Seleccionar y justificar una métrica secundaria apropiada para el problema
- **Requisito:** Proporcionar argumentación técnica sólida para la elección de la métrica
- **Ejemplos de métricas a considerar:** Precision, Recall, F1-Score, AUC-ROC

### 3. Ingeniería de Características
- **Objetivo:** Realizar feature engineering efectivo sobre datos JSON anidados
- **Requisito:** Documentar el proceso de selección y transformación de características

## Descripción Detallada de los Datos

### Formato del Dataset
- **Archivo:** `MLA_100k.jsonlines`
- **Formato:** JSON Lines (un objeto JSON válido por línea)
- **Tamaño:** 100,000 registros de productos de MercadoLibre Argentina (MLA)
- **Estructura:** Datos anidados que replican la respuesta de la API de MercadoLibre

### Estructura de Datos Típica

Cada registro contiene información compleja sobre un producto, incluyendo:

```json
{
  "id": "MLA123456789",
  "title": "iPhone 13 Pro Max 256gb",
  "condition": "new",  // Variable objetivo
  "price": 150000,
  "currency_id": "ARS",
  "initial_quantity": 5,
  "available_quantity": 3,
  "sold_quantity": 2,
  "listing_type_id": "gold_special",
  "start_time": "2023-01-15T10:30:00.000Z",
  "stop_time": "2023-04-15T10:30:00.000Z",
  "seller": {
    "id": 12345,
    "nickname": "vendedor123",
    "registration_date": "2020-01-01T00:00:00.000Z"
  },
  "seller_address": {
    "country": {"id": "AR", "name": "Argentina"},
    "state": {"id": "AR-B", "name": "Buenos Aires"},
    "city": {"id": "TUxBQkNBUzQzMjM", "name": "Capital Federal"}
  },
  "shipping": {
    "mode": "me2",
    "local_pick_up": true,
    "tags": ["fulfillment", "mandatory_free_shipping"]
  },
  "payment_methods": {
    "types": [
      {"id": "credit_card"},
      {"id": "bank_transfer"},
      {"id": "cash"}
    ]
  },
  "warranty": "Garantía del vendedor: 12 meses",
  // ... muchos más campos anidados
}
```

### Variables Clave Identificadas

**Variables Categóricas:**
- `condition` (variable objetivo): "new" o "used"
- `listing_type_id`: tipo de publicación (gold_special, gold_pro, etc.)
- `currency_id`: moneda de la transacción
- `seller_address`: ubicación del vendedor (país, estado, ciudad)
- `shipping.mode`: modalidad de envío
- `warranty`: información de garantía

**Variables Numéricas:**
- `price`: precio del producto
- `initial_quantity`: cantidad inicial publicada
- `available_quantity`: cantidad disponible
- `sold_quantity`: cantidad vendida
- Características temporales derivadas de `start_time` y `stop_time`

**Variables Booleanas Derivadas:**
- Métodos de pago disponibles (tarjeta de crédito, transferencia, efectivo)
- `shipping.local_pick_up`: retiro en persona
- `automatic_relist`: republicación automática

## Requisitos Técnicos

### Entorno de Desarrollo
```bash
# Configuración del entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencias Principales
- **Manipulación de datos:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Análisis:** matplotlib, seaborn
- **Desarrollo:** jupyter

### Función de Carga de Datos Proporcionada

```python
import json

def build_dataset():
    """
    Función proporcionada para cargar y dividir el dataset.
    
    Returns:
        X_train: Lista de diccionarios con datos de entrenamiento
        y_train: Lista de etiquetas de entrenamiento ("new" o "used")
        X_test: Lista de diccionarios con datos de prueba (sin etiquetas)
        y_test: Lista de etiquetas reales de prueba (para evaluación)
    """
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]  # Primeros 90,000 registros
    X_test = data[N:]   # Últimos 10,000 registros
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    
    # Eliminar la variable objetivo de los datos de prueba
    for x in X_test:
        del x["condition"]
    
    return X_train, y_train, X_test, y_test
```

## Entregables Requeridos

### 1. Código de la Solución
**Archivo principal:** Implementación completa del modelo de clasificación

**Debe incluir:**
- Preprocesamiento y limpieza de datos
- Ingeniería de características (feature engineering)
- Entrenamiento del modelo de Machine Learning
- Evaluación en datos de prueba
- Cálculo de métricas de rendimiento

**Estructura sugerida:**
```python
# Carga de datos
X_train, y_train, X_test, y_test = build_dataset()

# Preprocesamiento y feature engineering
def preprocess_data(X):
    # Implementar transformaciones
    pass

# Entrenamiento del modelo
def train_model(X_train_processed, y_train):
    # Implementar entrenamiento
    pass

# Evaluación del modelo
def evaluate_model(model, X_test_processed, y_test):
    # Implementar evaluación
    pass
```

### 2. Documento de Análisis Técnico

**Formato:** Markdown (.md) o Jupyter Notebook (.ipynb)

**Contenido obligatorio:**

#### 2.1 Análisis Exploratorio de Datos (EDA)
- Distribución de la variable objetivo
- Análisis de características más relevantes
- Identificación de patrones y anomalías
- Visualizaciones informativas

#### 2.2 Estrategia de Feature Engineering
- **Justificación de características seleccionadas**
- Proceso de transformación de datos anidados
- Manejo de valores faltantes
- Codificación de variables categóricas
- Creación de nuevas características derivadas

#### 2.3 Selección y Justificación de Métricas
- **Métrica principal:** Accuracy (justificar por qué es apropiada)
- **Métrica secundaria:** Selección y argumentación detallada
- Consideraciones sobre el balance de clases
- Interpretación de resultados en contexto de negocio

#### 2.4 Resultados y Rendimiento
- Rendimiento en conjunto de entrenamiento vs. prueba
- Análisis de la matriz de confusión
- Identificación de casos límite o errores comunes
- Comparación con diferentes algoritmos (opcional)

### 3. Análisis Exploratorio (Opcional)
- **Formato:** Jupyter Notebook (.ipynb)
- **Contenido:** EDA detallado con visualizaciones
- **Valor agregado:** Insights adicionales sobre los datos

## Criterios de Evaluación

### Criterios Técnicos (70%)

**Rendimiento del Modelo (30%)**
- ✅ Accuracy ≥ 0.86 en datos de prueba
- ✅ Métrica secundaria apropiada y bien justificada
- ✅ Análisis de resultados coherente

**Calidad del Código (25%)**
- ✅ Código limpio y bien documentado
- ✅ Uso apropiado de librerías de ML
- ✅ Manejo correcto de la división train/test
- ✅ Implementación robusta del preprocesamiento

**Feature Engineering (15%)**
- ✅ Transformación efectiva de datos JSON anidados
- ✅ Selección inteligente de características
- ✅ Justificación técnica de decisiones

### Criterios de Documentación (30%)

**Claridad y Completitud (20%)**
- ✅ Explicación clara del proceso de desarrollo
- ✅ Justificación de decisiones técnicas
- ✅ Documentación de resultados y conclusiones

**Análisis Crítico (10%)**
- ✅ Identificación de limitaciones del modelo
- ✅ Propuestas de mejora
- ✅ Consideraciones de implementación en producción

## Sugerencias de Implementación

### Pipeline de Desarrollo Recomendado

1. **Exploración inicial** (20% del tiempo)
   - Cargar y examinar estructura de datos
   - Análisis de distribución de clases
   - Identificación de campos relevantes

2. **Feature Engineering** (40% del tiempo)
   - Extracción de características de JSON anidado
   - Transformación de variables temporales
   - Codificación de variables categóricas
   - Creación de características derivadas

3. **Modelado** (30% del tiempo)
   - Experimentación con diferentes algoritmos
   - Validación cruzada para selección de hiperparámetros
   - Evaluación final en conjunto de prueba

4. **Documentación** (10% del tiempo)
   - Redacción del análisis técnico
   - Preparación de visualizaciones
   - Revisión y pulido final

### Algoritmos Sugeridos para Experimentar

- **XGBoost:** Excelente para datos tabulares mixtos
- **Random Forest:** Robusto y interpretable
- **Logistic Regression:** Baseline simple y efectivo
- **LightGBM:** Alternativa eficiente a XGBoost

### Consideraciones Especiales

**Manejo de Datos JSON Anidados:**
- Utilizar `pd.json_normalize()` para aplanar estructuras
- Extraer información relevante de objetos complejos
- Considerar técnicas de encoding para campos categóricos de alta cardinalidad

**Características Temporales:**
- Extraer información de `start_time` y `stop_time`
- Crear variables como duración de la publicación, día de la semana, mes
- Considerar características estacionales

**Validación:**
- Respetar la división temporal implícita en los datos
- No realizar data leakage entre conjuntos de entrenamiento y prueba
- Considerar validación cruzada temporal si es necesario

## Recursos Adicionales

### Documentación de la API de MercadoLibre
- [Documentación oficial](https://developers.mercadolibre.com/)
- Estructura de respuestas de la API de productos

### Tutoriales Recomendados
- Feature engineering para datos JSON
- XGBoost para clasificación binaria
- Interpretabilidad de modelos de ML

### Herramientas de Desarrollo
- Jupyter Notebook para experimentación
- pandas para manipulación de datos
- scikit-learn para ML pipeline
- matplotlib/seaborn para visualizaciones

---

**Tiempo estimado de desarrollo:** 8-12 horas  
**Fecha de entrega:** [A definir por el instructor]  
**Modalidad:** Individual  

**¡Buena suerte con el desarrollo de tu solución de clasificación de productos!**
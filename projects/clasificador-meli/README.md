# Clasificador de Productos Mercado Libre

Proyecto de Machine Learning para clasificar automáticamente si un producto de MercadoLibre es nuevo o usado basándose en sus características.

## Guía de Lectura para Estudiantes

### Por dónde empezar

1. **Lee primero:** `GUIA_LECTURA.md` - Explica la estructura del proyecto y qué hace cada archivo
2. **Entiende los datos:** `DICCIONARIO_DATOS.md` - Describe todas las variables del dataset
3. **Explora el código:** Comienza por los notebooks en `src/notebooks/`
4. **Prueba la aplicación:** Sigue las instrucciones de Uso Rápido más abajo

### Estructura del Proyecto

```
clasificador-meli/
├── src/                    # Código fuente
│   ├── features/           # Preprocesamiento de datos
│   ├── models/             # Entrenamiento del modelo (código)
│   ├── config/             # Configuraciones centralizadas
│   ├── utils/              # Funciones auxiliares
│   └── notebooks/          # Jupyter notebooks con EDA
├── models/                 # Modelos entrenados (artefactos .pkl)
├── ui/                     # Interfaz Streamlit
├── test_data/              # Datos de prueba
└── train.py                # Script principal para entrenar
```

**Nota importante:**
- `src/models/` = código Python para entrenar
- `models/` = archivos del modelo entrenado (.pkl)

Esta separación es una convención estándar en proyectos de ML.

## Descripción

Clasificador binario (new/used) que predice la condición de productos usando:
- Características del producto (precio, cantidad, vendedor)
- Métodos de pago disponibles
- Información de envío
- Variables temporales (fechas de publicación)

**Modelo:** XGBoost
**Accuracy:** ~86%
**Dataset:** 100,000 productos de MercadoLibre Argentina

---

## Instalación

### Requisitos

- Python 3.8+
- Git LFS (para descargar el dataset)

### Paso 1: Clonar repositorio

```bash
git clone https://github.com/dpalacioj/data-projects-lab.git
cd data-projects-lab/projects/clasificador-meli
```

### Paso 2: Instalar Git LFS (si no lo tienes)

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Linux:**
```bash
sudo apt-get install git-lfs
git lfs install
```

**Windows:**
Descarga desde https://git-lfs.github.com/

### Paso 3: Descargar dataset

```bash
git lfs pull
```

Verifica que el archivo existe:
```bash
ls -lh ../../datasets/MLA_100k.jsonlines
# Debe mostrar ~316 MB
```

### Paso 4: Instalar dependencias

**Opción A: Con uv (recomendado)**
```bash
# Desde la raíz del repositorio data-projects-lab/
uv sync
```
---

## Uso Rápido

### 1. Entrenar el Modelo

```bash
# Entrenar con configuración base
python train.py

# Entrenar con optimización de hiperparámetros
python train.py --optimize

# Forzar reprocesamiento desde JSON (ignora parquet)
python train.py --from-json
```

El modelo se guardará en `models/xgb_model_v1.pkl`

### 2. Ejecutar UI de Streamlit

```bash
streamlit run ui/streamlit_app.py
```

La aplicación se abrirá en http://localhost:8501

### 3. Probar con Datos de Ejemplo

En la UI de Streamlit, puedes cargar archivos de `test_data/`:

- `test_products_complete.csv` - CSV con 5 productos
- `test_products.json` - JSON con 5 productos
- `test_products.jsonlines` - JSONLINES con 5 productos
- `single_product_example.json` - Ejemplo para ingreso manual
- `test_products_processed.parquet`- Archivo parquet

---

## Características de la UI

La interfaz de Streamlit ofrece:

1. **5 opciones de carga de datos:**
   - CSV
   - JSON
   - JSONLINES
   - Parquet
   - Ingreso Manual (copiar/pegar JSON)

2. **Visualización estandarizada:**
   - Métricas de resumen (Total, NEW, USED con porcentajes)
   - Gráfico de barras de distribución
   - Tabla detallada con gradient de confianza
   - Estadísticas adicionales por condición

3. **Probabilidades de predicción:**
   - `probability_new`: Probabilidad de ser nuevo
   - `probability_used`: Probabilidad de ser usado
   - `confidence`: Confianza de la predicción (max probability)

4. **Descarga de resultados:**
   - CSV, JSON o Parquet según el formato de entrada

---

## Dataset

**Nombre:** MLA_100k.jsonlines
**Ubicación:** `../../datasets/MLA_100k.jsonlines`
**Tamaño:** 316 MB
**Registros:** ~100,000 productos
**Formato:** JSON Lines (un JSON por línea)

### Ejemplo de registro

```json
{
  "title": "Auriculares Samsung...",
  "condition": "new",
  "price": 80,
  "seller_id": 74952096,
  "listing_type_id": "bronze",
  "seller_address": {...},
  "shipping": {...},
  "...": "..."
}
```

Ver `DICCIONARIO_DATOS.md` para descripción completa de todas las variables.

---

## Flujo de Trabajo

### 1. Entrenamiento

```
datasets/MLA_100k.jsonlines
    ↓
src/features/preprocessing.py (transform)
    ↓
src/models/train_model.py (train)
    ↓
models/xgb_model_v1.pkl (guardado)
```

### 2. Predicción (UI)

```
Usuario sube archivo
    ↓
ui/streamlit_app.py (carga datos)
    ↓
src/features/preprocessing.py (transform)
    ↓
models/xgb_model_v1.pkl (predict)
    ↓
ui/streamlit_app.py (display_results)
```

---

## Sistema de Logging

El proyecto incluye un sistema de logging centralizado que registra eventos importantes:

```python
from src.utils.logger import setup_logger
logger = setup_logger(__name__)

logger.info("Iniciando entrenamiento...")
logger.debug("Detalles de depuración...")
logger.error("Error encontrado", exc_info=True)
```

Los logs se muestran en consola con formato:
```
2025-10-26 14:32:15 | INFO | preprocessing | Iniciando preprocesamiento | Shape: (100000, 48)
```

Para probar el sistema de logging:
```bash
python test_logging.py
```

---

## Archivos Importantes

- `GUIA_LECTURA.md` - Guía detallada de lectura del proyecto
- `DICCIONARIO_DATOS.md` - Descripción de variables del dataset
- `train.py` - Script principal de entrenamiento
- `test_logging.py` - Script de prueba del sistema de logging
- `test_data/README.md` - Instrucciones para usar datos de prueba

---

## Tecnologías Utilizadas

- **Python 3.8+**
- **XGBoost** - Modelo de clasificación
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Preprocesamiento y métricas
- **Streamlit** - Interfaz de usuario interactiva
- **Joblib** - Serialización del modelo
- **Git LFS** - Versionado de archivos grandes

---

## Métricas del Modelo

**Modelo entrenado:** XGBoost Classifier
**Accuracy:** ~86%
**Clases:** Binary (new, used)

El modelo fue entrenado con:
- 80% datos de entrenamiento
- 20% datos de prueba
- Validación cruzada en optimización
- Grid Search para hiperparámetros

---

## Preguntas Frecuentes

**¿Por qué hay dos carpetas models/?**
- `src/models/` = código Python (train_model.py, xgb_model.py)
- `models/` = archivos del modelo entrenado (.pkl, .json)

Esta es una convención estándar en proyectos de ML.

**¿Qué es el archivo .parquet en src/data/?**
Es el dataset preprocesado guardado en formato Parquet (más rápido de cargar que JSON). Se genera automáticamente al entrenar.

**¿Cómo agrego nuevos datos?**
1. Agrega productos al dataset JSON
2. Ejecuta `python train.py --from-json` para reentrenar

**¿Puedo usar otro modelo?**
Sí. Modifica `src/models/train_model.py` para usar otro clasificador de scikit-learn o XGBoost.

---

## Licencia

MIT License - Ver archivo LICENSE en la raíz del repositorio

---

**Desarrollado para fines educativos**
Data Projects Lab - 2025

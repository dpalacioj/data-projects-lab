---
title: "02 · Clasificador Mercado Libre"
parent: Proyectos
nav_order: 2
---
# Clasificador de Productos Mercado Libre

Proyecto de Machine Learning para clasificar automáticamente si un producto de MercadoLibre es nuevo o usado basándose en sus características.

## En una línea

Clasificador binario (**nuevo** vs. **usado**) para publicaciones de
MercadoLibre Argentina, entrenado con XGBoost sobre 100 000 productos.
Accuracy ≈ 86 %, ROC-AUC ≈ 93,6 %.

## Qué hace el proyecto

Dado un producto con su título, precio, vendedor, métodos de pago,
información de envío y fechas de publicación, decide si es un
**artículo nuevo** o un **artículo usado**. Es un ejemplo de
clasificación tabular con ingeniería de variables manual (21 features
derivadas) y XGBoost como modelo.

Lo interesante pedagógicamente no es el modelo — es ver **cómo se parte
de un JSON Lines con campos anidados y se llega a una tabla modelable**.

## Estructura del proyecto

```
02-clasificador-meli/
├── src/                     # Código fuente
│   ├── features/            # preprocessing.py (21 features)
│   ├── models/              # train_model.py, xgb_model.py
│   ├── config/              # config.py (rutas, hiperparámetros)
│   ├── utils/               # logger.py
│   └── notebooks/           # EDA (preprocessing_eda.ipynb) + XGBoost (model_xgboost.ipynb)
├── models/                  # Artefacto entrenado (xgb_model_v1.pkl)
├── ui/                      # streamlit_app.py
├── test_data/               # 5 productos de ejemplo (CSV/JSON/Parquet)
├── train.py                 # Entrada CLI para entrenar
├── GUIA_LECTURA.md          # Orden sugerido de archivos para leer el código
└── DICCIONARIO_DATOS.md     # Descripción de cada campo del dataset
```

> **Dos carpetas `models/` — ¿por qué?**
> `src/models/` guarda el **código** que define y entrena el modelo
> (clases Python). `models/` guarda el **artefacto** entrenado (`.pkl`).
> Es la convención del repo: código versus artefactos, en carpetas
> distintas.

## Variables del dataset (resumen)

| Grupo | Columnas de ejemplo | Qué aportan |
|---|---|---|
| Producto | `title`, `price`, `initial_quantity`, `category_id` | Características intrínsecas |
| Vendedor | `seller_id`, `seller_country`, `seller_state`, `seller_city` | Reputación geográfica |
| Métodos de pago | `accepts_mercadopago`, columnas one-hot por método | Señal de confianza |
| Envío | `shipping_free_shipping`, `shipping_mode` | Formato comercial |
| Temporales | `year_start`, `month_start`, `days_active`, `week_day` | Antigüedad y estacionalidad |
| Garantía | `garantia_aplica` (derivada por regex del campo `warranty`) | Nuevo vs. usado tiende a correlacionar |

> El detalle de cada variable (tipo, valores, nulos) está en
> [`DICCIONARIO_DATOS.md`](DICCIONARIO_DATOS.md). El orden sugerido para
> leer el código está en [`GUIA_LECTURA.md`](GUIA_LECTURA.md).

---

## Instalación

### Requisitos

- Python 3.11 (igual que el resto del curso)
- Git LFS (para descargar el dataset de ~316 MB)

### Paso 1: Clonar repositorio

```bash
git clone https://github.com/dpalacioj/data-projects-lab.git
cd data-projects-lab
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
uv sync --extra meli --extra dev
```

El extra `meli` trae `xgboost`, `pydantic`, `joblib` y `scipy`. El extra
`dev` trae `jupyter`, `pytest` y `ruff`.
---

## Uso Rápido

Todos los comandos de abajo asumen que estás parado en la carpeta del
proyecto: `cd projects/02-clasificador-meli` desde la raíz del repo.

### 1. Entrenar el Modelo

```bash
# Entrenar con configuración base
uv run python train.py

# Entrenar con optimización de hiperparámetros
uv run python train.py --optimize

# Forzar reprocesamiento desde JSON (ignora parquet)
uv run python train.py --from-json
```

El modelo se guardará en `models/xgb_model_v1.pkl`.

### 2. Ejecutar UI de Streamlit

```bash
uv run streamlit run ui/streamlit_app.py
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
uv run python test_logging.py
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

- **Python 3.11**
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

**¿Qué es el archivo .parquet en src/data/?**
Es el dataset preprocesado guardado en formato Parquet (más rápido de cargar que JSON). Se genera automáticamente al entrenar.

**¿Cómo agrego nuevos datos?**
1. Agrega productos al dataset JSON
2. Ejecuta `uv run python train.py --from-json` para reentrenar

**¿Puedo usar otro modelo?**
Sí. Modifica `src/models/train_model.py` para usar otro clasificador de scikit-learn o XGBoost.

---

## Licencia

MIT License — ver `LICENSE` en la raíz del repositorio.

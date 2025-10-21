# Guía de Lectura: Clasificador MeLi

## Objetivo
Clasificar productos de MercadoLibre como **nuevo** o **usado** usando XGBoost.

**Dataset**: 100k productos (316 MB) | **Modelo**: XGBoost | **Accuracy**: ~86% | **ROC-AUC**: ~93.6%

---

## Orden de Lectura

### 1. Contexto (10 min)
```
problema-a-resolver.md → README.md
```

### 2. Exploración de Datos (60 min)
```
src/notebooks/preprocessing_eda.ipynb    [★ EMPIEZA AQUÍ]
```
**Qué aprenderás:**
- Estructura del dataset (JSON Lines, campos anidados)
- Calidad de datos (missing values, distribuciones)
- Exploración inicial con archivo pequeño (celdas 1-6)
- Análisis completo del dataset (celda 8 en adelante)

**Por qué las celdas 1-6 no cargan todos los datos:**
Son para exploración rápida con un CSV pequeño de prueba antes de cargar los 316 MB completos. El dataset real se carga en la **celda 8**.

### 3. Feature Engineering (45 min)
```
src/features/preprocessing.py
src/config/config.py
```
**21 features creadas:**
- **Temporales**: `year_start`, `month_start`, `days_active`, `week_day`
- **Geográficas**: `seller_country`, `seller_state`, `seller_city`
- **Booleanas**: `shipping_free_shipping`, `accepts_mercadopago`, `garantia_aplica`
- **One-hot**: Métodos de pago, tags

**Código clave:**
- `clasificar_garantia()`: Regex para clasificar warranty
- `extract_payment_methods()`: One-hot encoding de métodos de pago
- `transform()`: Pipeline completo

### 4. Modelado (90 min)
```
src/notebooks/model_xgboost.ipynb       [Experimentación]
src/models/xgb_model.py                 [Implementación]
src/models/train_model.py               [Pipeline]
```
**Flujo:**
1. Split estratificado 80/20
2. GridSearchCV para hiperparámetros
3. Entrenamiento XGBoost
4. Evaluación: Accuracy, Precision, Recall, ROC-AUC
5. Análisis de feature importance

**Ejecutar:**
```bash
python train.py              # Entrenamiento base
python train.py --optimize   # Con optimización
```

### 5. Despliegue (30 min)
```
ui/streamlit_app.py
```
**Modos de input:**
- CSV con múltiples productos
- JSON Lines
- JSON individual (pegar en textarea)

**Ejecutar:**
```bash
streamlit run ui/streamlit_app.py
```

---

## Resumen de Archivos

```
clasificador-meli/
├── problema-a-resolver.md              # 1. Contexto
├── src/notebooks/
│   ├── preprocessing_eda.ipynb         # 2. EDA [EMPIEZA AQUÍ]
│   └── model_xgboost.ipynb             # 4. Experimentación
├── src/config/config.py                # Rutas centralizadas
├── src/features/preprocessing.py       # 3. Pipeline de transformación
├── src/models/
│   ├── xgb_model.py                    # Clase XGBoost
│   └── train_model.py                  # Pipeline de training
├── train.py                            # Entry point
└── ui/streamlit_app.py                 # 5. Demo interactiva
```

---

## Actividades Prácticas

### 1. Nueva Feature
Agrega en `preprocessing.py`:
```python
def extract_brand_from_title(self, title):
    brands = ["Samsung", "Apple", "LG", "Sony"]
    for brand in brands:
        if brand.lower() in title.lower():
            return brand
    return "Other"
```

### 2. Interpretabilidad con SHAP
En `model_xgboost.ipynb`:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### 3. API REST con FastAPI
Crea `api/main.py`:
```python
from fastapi import FastAPI
from joblib import load

app = FastAPI()
model = load("../models/xgb_model_v1.pkl")

@app.post("/predict")
def predict(product: dict):
    # Preprocesar y predecir
    return {"condition": "new" or "used"}
```

---

## Troubleshooting

**"FileNotFoundError: datos/MLA_100k.jsonlines"**
```bash
git lfs pull
```

**"ModuleNotFoundError: No module named 'src'"**
```bash
cd projects/clasificador-meli
python train.py
```

---

## Checklist de Aprendizaje

- [ ] Cargar y explorar JSON Lines
- [ ] Manejar valores faltantes y campos anidados
- [ ] Crear features temporales y one-hot encoding
- [ ] Entrenar XGBoost y optimizar hiperparámetros
- [ ] Evaluar con múltiples métricas
- [ ] Interpretar feature importance
- [ ] Crear interfaz con Streamlit

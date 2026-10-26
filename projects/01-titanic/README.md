---
title: "01 · Titanic"
parent: Proyectos
nav_order: 1
---
# 🚢 Proyecto Titanic: Predicción de Supervivencia

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)

Proyecto completo de Machine Learning end-to-end para predecir la supervivencia de pasajeros del Titanic. Incluye exploración de datos, entrenamiento con AutoML, interpretabilidad de modelos y deployment con Streamlit.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Pipeline del Proyecto](#-pipeline-del-proyecto)
- [Instalación y Uso](#-instalación-y-uso)
- [Resultados y Métricas](#-resultados-y-métricas)
- [Interpretabilidad](#-interpretabilidad)
- [Deployment](#-deployment)
- [Tecnologías](#-tecnologías)

---

## 📖 Descripción

Este proyecto demuestra un **flujo completo de Machine Learning** utilizando el famoso dataset del Titanic. El objetivo es predecir si un pasajero sobrevivió al desastre basándose en características como edad, género, clase, tarifa, y número de familiares a bordo.

### ¿Qué Aprenderás?

- ✅ Descarga y limpieza de datos desde OpenML
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Preprocesamiento con pipelines de scikit-learn
- ✅ Comparación de modelos con AutoML (PyCaret y FLAML)
- ✅ Optimización de hiperparámetros con GridSearchCV
- ✅ Interpretación de modelos con Feature Importance
- ✅ Deployment de modelos con Streamlit

### Dataset

- **Fuente:** OpenML ([Dataset 1309](https://www.openml.org/d/1309))
- **Tamaño:** 1,309 pasajeros
- **Características:** 9 variables (después de limpieza)
- **Target:** `survived` (0 = No sobrevivió, 1 = Sobrevivió)

---

## 📁 Estructura del Proyecto

```
projects/01-titanic/
├── 01_descarga_datos.ipynb                    # Descarga y preparación inicial
├── 02_ajuste_datos.ipynb                      # Limpieza y transformación
├── 04_AutoML_PyCaret.ipynb                    # Exploración con PyCaret
├── 05_AutoML_FLAML.ipynb                      # Exploración con FLAML
├── 06_entrenamiento_modelo.ipynb              # Entrenamiento y optimización
├── 07_interpretacion_modelos.ipynb            # Interpretabilidad
├── 08_titanic_streamlit.py                    # App de deployment
├── modelos/
│   ├── gbc_tuned_model.pkl                    # Modelo GBC (PyCaret)
│   ├── gbc_tuned_model.joblib                 # Modelo GBC (joblib)
│   └── titanic_classification-random_forest-v1.joblib  # Modelo final
└── imagenes/
    └── titanic.jpg                            # Imagen para Streamlit
```

---

## 🔄 Pipeline del Proyecto

### 1️⃣ Descarga de Datos (`01_descarga_datos.ipynb`)

**Objetivo:** Obtener el dataset inicial desde OpenML

- Descarga 1,309 registros con 14 columnas
- Identifica y elimina columnas con **data leakage** (`boat`, `body`)
- Selecciona 12 características relevantes
- Exporta en formato **Parquet** para eficiencia

**Características Iniciales:**
- `pclass`, `survived`, `name`, `sex`, `age`, `sibsp`, `parch`, `ticket`, `fare`, `cabin`, `embarked`, `home.dest`

---

### 2️⃣ Ajuste de Datos (`02_ajuste_datos.ipynb`)

**Objetivo:** Limpieza y preparación para modelado

**Transformaciones:**
- Reemplaza placeholders "?" por `NaN`
- Elimina columnas con alta proporción de nulos: `cabin`, `ticket`, `home.dest`
- Convierte tipos de datos:
  - **Categóricas ordinales:** `pclass` (1, 2, 3)
  - **Categóricas nominales:** `sex`, `embarked`
  - **Numéricas continuas:** `age`, `fare`
  - **Numéricas discretas:** `sibsp`, `parch`
  - **Booleana:** `survived`

**Dataset Final:** 1,309 filas × 9 columnas

---

### 3️⃣ AutoML con PyCaret (`04_AutoML_PyCaret.ipynb`)

**Objetivo:** Exploración rápida de múltiples modelos

**Proceso:**
1. Setup automático con preprocesamiento integrado
2. Comparación de 4 algoritmos: LogisticRegression, RandomForest, GBC, LightGBM
3. Fine-tuning del mejor modelo (Gradient Boosting Classifier)

**Resultados:**
- Mejor modelo: **Gradient Boosting Classifier**
- Accuracy: ~82% en validación interna
- Modelo guardado: `gbc_tuned_model.pkl`

---

### 4️⃣ AutoML con FLAML (`05_AutoML_FLAML.ipynb`)

**Objetivo:** Optimización eficiente con presupuesto de tiempo

**Diferencias vs PyCaret:**
- Preprocesamiento manual con `ColumnTransformer`
- Búsqueda económica de hiperparámetros (CFO algorithm)
- Mayor control y transparencia

**Proceso:**
1. Split 80/20 (1,047 entrenamiento / 262 prueba)
2. Pipeline de preprocesamiento personalizado
3. Búsqueda automática en 60 segundos
4. Modelos evaluados: LGBM, RF, XGBoost, ExtraTrees, LogisticRegression

**Resultados:**
- Mejor modelo: **LightGBM**
- Accuracy test: **84.35%**
- Tiempo de búsqueda: ~21 segundos

---

### 5️⃣ Entrenamiento Final (`06_entrenamiento_modelo.ipynb`)

**Objetivo:** Seleccionar, optimizar y guardar el modelo de producción

**Modelos Evaluados:**

| Pipeline | Accuracy | F1 | ROC-AUC |
|----------|----------|-----|---------|
| **RandomForest_median** | **0.7489** | **0.667** | **0.728** |
| LogisticRegression_mean_scale | 0.7444 | 0.689 | 0.735 |
| LogisticRegression_median | 0.7354 | 0.681 | 0.727 |
| RandomForest_mean_scale | 0.7265 | 0.643 | 0.707 |

**Modelo Seleccionado:** RandomForest con imputación por mediana

**Razones:**
- ✅ Robustez contra outliers
- ✅ Consistencia en validación cruzada (5-folds)
- ✅ Simplicidad e interpretabilidad
- ✅ Mejor generalización

**Hiperparámetros Optimizados (GridSearchCV):**
```python
{
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 10
}
```

**Modelo Guardado:** `titanic_classification-random_forest-v1.joblib` (754 KB)

---

### 6️⃣ Interpretación (`07_interpretacion_modelos.ipynb`)

**Objetivo:** Entender qué aprende el modelo

**Feature Importance (Top 7):**

| Variable | Importancia | Descripción |
|----------|-------------|-------------|
| **fare** | 21.3% | Tarifa del boleto (proxy de clase socioeconómica) |
| **age** | 19.2% | Edad del pasajero |
| **sex_female** | 19.1% | Género femenino |
| **sex_male** | 17.1% | Género masculino |
| **pclass** | 12.5% | Clase del boleto (1ª, 2ª, 3ª) |
| **parch** | 4.3% | Número de padres/hijos a bordo |
| **sibsp** | 3.5% | Número de hermanos/cónyuge a bordo |

**Insights Clave:**
- 💰 **Tarifa y clase** son los factores más influyentes (refleja desigualdad socioeconómica)
- 👥 **Género** es crítico (política "mujeres y niños primero")
- 📊 **Edad** también importante (niños fueron priorizados)
- 👨‍👩‍👧 **Familiares** tienen menor impacto

---

### 7️⃣ Deployment (`08_titanic_streamlit.py`)

**Objetivo:** App web interactiva para predicciones en tiempo real

**Características:**
- Interfaz intuitiva con Streamlit
- Entrada de 7 características del pasajero
- Predicción instantánea con modelo cargado
- Visualización con imagen histórica del Titanic

**Inputs del Usuario:**
```
Numéricos:
  - Age: 0-100 años
  - Fare: 0-300 (costo del boleto)
  - SibSp: 0-15 (hermanos/cónyuge)
  - Parch: 0-15 (padres/hijos)

Categóricos:
  - Pclass: 1st / 2nd / 3rd
  - Sex: Woman / Man
  - Embarked: Cherbourg / Queenstown / Southampton
```

---

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recomendado)
- Jupyter Notebook/Lab

### Instalación

```bash
# Desde la raíz del repositorio
cd data-projects-lab

# Instalar dependencias base
uv sync

# Instalar dependencias opcionales (AutoML, interpretabilidad)
uv pip install -e ".[all]"
```

### Ejecutar Notebooks

```bash
# Iniciar Jupyter Lab
uv run jupyter lab

# Navegar a projects/01-titanic/
# Ejecutar en orden: 01 → 02 → 06 → 07
# Los AutoML (04 y 05) requieren un entorno aparte, ver seccion abajo.
```

### Ejecutar 04_AutoML_PyCaret y 05_AutoML_FLAML en un entorno aislado

**Por que**: el `.venv` del monorepo tiene `scikit-learn >= 1.9` porque
el proyecto 00 lo exige. **PyCaret** depende de `sklearn.utils._print_elapsed_time`
(API privada removida en sklearn 1.6) y **FLAML 2.3.x** aun no implementa
`__sklearn_tags__` en `LGBMEstimator`, requerido por sklearn 1.6+.
Resultado: ambos cuadernos fallan al importar en el `.venv` compartido.

**Que hacer**: crear un venv aparte, solo para estos dos cuadernos, con
`scikit-learn==1.5.*`. No hace falta tocar nada del monorepo.

```bash
# Desde la raiz del repo (data-projects-lab/):
cd projects/01-titanic

# 1. Crear un venv dedicado con uv (python 3.11)
uv venv .venv-automl --python 3.11

# 2. Activarlo
source .venv-automl/bin/activate        # macOS/Linux
# .venv-automl\Scripts\activate         # Windows

# 3. Instalar las deps compatibles
uv pip install "scikit-learn==1.5.*" "pandas>=2.1" "numpy>=1.26" \
               "matplotlib>=3.8" "jupyter" "ipykernel" \
               "pycaret>=3.3" "flaml[automl]>=2.3.6" "plotly>=5.5"

# 4. Registrar este venv como un kernel de Jupyter visible desde el .venv del monorepo
python -m ipykernel install --user --name titanic-automl --display-name "Python (titanic-automl)"

# 5. Desactivar y volver al .venv del monorepo
deactivate
```

Despues, en Jupyter: abrir `04_AutoML_PyCaret.ipynb` o `05_AutoML_FLAML.ipynb`,
y en el menu **Kernel → Change Kernel** seleccionar
**Python (titanic-automl)**. Ejecutar normalmente.

> El venv `.venv-automl` queda ignorado por git gracias al patron `.venv*`
> del `.gitignore`. Si en el futuro PyCaret y FLAML se actualizan para
> sklearn >= 1.6, este paso deja de ser necesario.

### Ejecutar Streamlit App

```bash
# Opción 1: Con uv (recomendado)
uv run streamlit run projects/01-titanic/08_titanic_streamlit.py

# Opción 2: Con entorno virtual activado
source .venv/bin/activate
streamlit run projects/01-titanic/08_titanic_streamlit.py

# La app se abrirá en http://localhost:8501
```

### Usar el Modelo en Python

```python
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load("projects/01-titanic/modelos/titanic_classification-random_forest-v1.joblib")

# Crear datos de entrada
data = pd.DataFrame({
    'age': [25],
    'fare': [50.0],
    'sibsp': [1],
    'parch': [0],
    'pclass': [2],
    'sex': ['female'],
    'embarked': ['C']
})

# Predecir
prediction = model.predict(data)
print(f"Sobrevive: {'Sí' if prediction[0] == 1 else 'No'}")
```

---

## 📊 Resultados y Métricas

### Modelo Final: RandomForest_median

**Validación Cruzada (5-folds):**
```
Media Accuracy: 80.36%
  - Fold 1: 70.95%
  - Fold 2: 75.84%
  - Fold 3: 79.78%
  - Fold 4: 77.53%
  - Fold 5: 78.09%
```

**Test Set:**
```
Accuracy:  74.89%
Precision: 74.67%
Recall:    60.22%
F1-Score:  66.67%
ROC-AUC:   0.728
```

**Interpretación:**
- El modelo tiene **buena precisión** (evita falsos positivos)
- **Recall moderado** (más conservador en predicciones)
- **Generaliza bien** con varianza baja entre folds

---

## 🔍 Interpretabilidad

### Factores Clave de Supervivencia

1. **Tarifa del Boleto (21.3%):** Proxy de riqueza y acceso a mejores ubicaciones
2. **Edad (19.2%):** Niños fueron priorizados en evacuación
3. **Género (36.2% combinado):** Política "mujeres y niños primero"
4. **Clase (12.5%):** Primera clase tuvo mejor acceso a botes salvavidas

### Patrones Aprendidos

El modelo refleja la **desigualdad socioeconómica** y las **políticas de evacuación** del Titanic:
- Pasajeros de primera clase (tarifa alta) tuvieron mejor supervivencia
- Mujeres y niños fueron priorizados
- Familias con hijos (parch > 0) tuvieron ligera ventaja

---

## 🚀 Deployment

### Arquitectura Streamlit

```
Usuario → Streamlit UI → get_user_data() → DataFrame
                             ↓
                   load_model() (cached)
                             ↓
              Pipeline.predict(user_data)
                             ↓
         Preprocesamiento + RandomForest
                             ↓
                Predicción: 0 o 1
                             ↓
              Muestra resultado con emoji
```

### Tecnologías de Deployment

- **Streamlit:** Framework web interactivo
- **Joblib:** Serialización eficiente de modelos
- **scikit-learn Pipeline:** Preprocesamiento automático

---

## 🛠️ Tecnologías

### Librerías Principales

| Categoría | Herramientas |
|-----------|-------------|
| **Data Science** | pandas, numpy, scikit-learn |
| **Visualización** | matplotlib, seaborn, plotly |
| **AutoML** | PyCaret (>=3.3.0), FLAML (>=2.3.6) |
| **Deployment** | Streamlit (>=1.50.0) |
| **Serialización** | joblib, pickle |
| **Formato de Datos** | PyArrow (Parquet) |

### Python Version

- **Python:** 3.11

---

## 🎓 Valor Educativo

Este proyecto es ideal para aprender:

1. **Flujo Completo de ML:** Desde datos crudos hasta modelo en producción
2. **AutoML vs Manual:** Comparación entre PyCaret, FLAML y scikit-learn
3. **Mejores Prácticas:** Pipelines, validación cruzada, test/train split
4. **Interpretabilidad:** Entender qué aprende el modelo
5. **Deployment:** Poner modelos en producción con Streamlit

---

## 📝 Próximos Pasos

Posibles extensiones del proyecto:

- [ ] Agregar técnicas de interpretabilidad avanzadas (SHAP, LIME)
- [ ] Implementar Feature Engineering adicional
- [ ] Probar modelos de ensemble (stacking, blending)
- [ ] Agregar análisis de error detallado
- [ ] Deploy en Streamlit Cloud o Hugging Face Spaces
- [ ] Crear API REST con FastAPI

---

## 👤 Autor

**David Palacio Jiménez**

- 📧 Email: davidpalacioj@gmail.com
- 🐙 GitHub: [dpalacioj](https://github.com/dpalacioj)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](../../LICENSE) para más detalles.

**Copyright (c) 2025 David Palacio Jiménez**

---

⭐️ **Si este proyecto te resulta útil, considera darle una estrella al repositorio!**

🚀 **¡Feliz aprendizaje!**

---
title: "D1 · Preprocesamiento"
parent: Proyectos
nav_order: 10
---
# Ejercicio: Comparación de Técnicas de Preprocesamiento

## Descripción

Ejercicio de clase donde el estudiante debe **comparar diferentes técnicas
de preprocesamiento** (imputación, codificación de categóricas, escalado,
selección de features) y medir empíricamente cómo cambian las métricas
de un modelo de clasificación.

El objetivo es que el estudiante **descubra por sí mismo** que:

- Algunas decisiones de preprocesamiento cambian las métricas drásticamente.
- El efecto depende del modelo: los **modelos lineales** son mucho más
  sensibles al preprocesamiento que los **modelos basados en árboles**.
- No existe una "mejor receta" universal — depende del dataset y del
  modelo.

## Objetivo de Aprendizaje

Al terminar el ejercicio, el estudiante debe ser capaz de:

1. Diseñar varios pipelines de preprocesamiento usando `Pipeline` y
   `ColumnTransformer` de scikit-learn.
2. Comparar técnicas de imputación: **drop**, **mediana/moda**, **categoría "Unknown"**.
3. Comparar codificación de categóricas: **OneHot**, **Ordinal**, eliminar.
4. Comparar escaladores: **StandardScaler**, **MinMaxScaler**, **log1p**.
5. Medir resultados con métricas (Accuracy, F1, ROC-AUC) en un mismo conjunto
   de prueba para hacer la comparación justa.
6. Justificar la elección final con argumentos basados en datos.

## Dataset: Adult Census Income (UCI)

- **Fuente:** https://archive.ics.uci.edu/dataset/2/adult
- **Tarea:** clasificación binaria (`income > 50K USD/año` o no).
- **Filas:** ~48,800.
- **Variables:** 14 (mezcla numéricas, categóricas, ordinales).
- **Faltantes:** ~7% en `workclass`, `occupation` y `native-country`
  (codificados como `?`).
- **Licencia:** Creative Commons CC BY 4.0.

### ¿Por qué este dataset?

Tiene **todas** las complicaciones que un estudiante debe aprender a manejar:

| Característica | Por qué importa |
|---|---|
| Faltantes en categóricas | Probar drop / moda / "Unknown". |
| Variable ordinal (`education-num`) | Decidir si tratarla como numérica u ordinal. |
| Categórica de **alta cardinalidad** (`native-country`, ~40 valores) | OHE explota la dimensión; ¿valdrá la pena? |
| Variables numéricas **muy sesgadas** (`capital-gain`, `capital-loss`) | Probar `log1p` y diferentes escaladores. |
| Class imbalance (~24% de positivos) | Importancia de F1 sobre accuracy. |

## Estructura

```
ejercicio-preprocesamiento/
├── README.md                                  # Este archivo
├── 01_ejercicio_preprocesamiento.ipynb        # Plantilla con TODOs
└── 02_solucion_preprocesamiento.ipynb         # Solución comentada
```

## Cómo Usar

### Estudiantes

1. Abre `01_ejercicio_preprocesamiento.ipynb`.
2. Lee el contexto y el enunciado.
3. Completa los `TODO` siguiendo la guía paso a paso.
4. Intenta responder las preguntas de reflexión sin mirar la solución.
5. Compara con `02_solucion_preprocesamiento.ipynb`.

### Profesores

- El ejercicio está pensado para una sesión de **2 horas**.
- Puedes pedirles que entreguen la tabla de comparación final + un
  párrafo justificando la mejor configuración.
- Variantes posibles para evaluar:
  - Cambiar el dataset (German Credit, Heart Disease).
  - Pedir que agreguen un escalador robusto (`RobustScaler`).
  - Pedir feature selection adicional (`SelectKBest`).

## Requisitos

```bash
# Desde la raíz del repositorio
uv sync
```

Las dependencias base (pandas, numpy, scikit-learn, matplotlib, seaborn,
jupyter) son suficientes para este ejercicio. No requiere extras.

## Ejecución

```bash
cd projects/d1-ejercicio-preprocesamiento
jupyter lab 01_ejercicio_preprocesamiento.ipynb
```

## Resultados Esperados

Tras completar el ejercicio, el estudiante debe ver algo así
(números aproximados, varían un poco entre ejecuciones):

| Estrategia | Modelo | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Drop NaN + OHE | LogReg | 0.843 | 0.655 | 0.892 |
| Imputar + OHE + Standard | LogReg | 0.850 | 0.654 | 0.903 |
| Imputar + OHE + log1p | LogReg | 0.844 | 0.642 | 0.898 |
| Imputar + OHE + MinMax | LogReg | 0.850 | 0.652 | 0.901 |
| Imputar + Ordinal | LogReg | **0.824** | **0.550** | **0.851** |
| Imputar + OHE sin alta-card | LogReg | 0.850 | 0.653 | 0.902 |
| ... mismas estrategias con RandomForest | RF | ~0.85 | ~0.66 | ~0.90 |

**Spread esperado:**
- LogReg: hasta **10 puntos de F1** entre estrategias.
- Random Forest: <1 punto en cualquier métrica.

> El estudiante debe explicar **por qué** los árboles son tan robustos
> frente a las decisiones de preprocesamiento, mientras que los modelos
> lineales son tan sensibles.

## Conceptos Clave

- `SimpleImputer` con `strategy="median"`, `"most_frequent"`, `"constant"`.
- `OneHotEncoder` con `handle_unknown="ignore"`.
- `OrdinalEncoder` y por qué es peligroso con modelos lineales.
- `StandardScaler` vs `MinMaxScaler` vs `log1p` + escalador.
- `ColumnTransformer` y `Pipeline` para encadenar transformaciones.
- Cuidado con el **data leakage**: ajustar el imputer/escalador
  **solo en el train**.

## Referencias

- [Scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn: Compose](https://scikit-learn.org/stable/modules/compose.html)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.

---

**Licencia:** MIT
**Autor:** David Palacio Jiménez · davidpalacioj@gmail.com

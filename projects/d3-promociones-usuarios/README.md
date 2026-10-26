---
title: "D3 · Target de promociones"
parent: Proyectos
nav_order: 12
---
# d3 · Construcción de target: ¿a quién le damos una promoción?

> Demo corta (~20 min). Cubre un solo concepto: **cómo se construye la
> variable objetivo (`y`) cuando el dataset no la trae marcada**.

## 📚 Descripción

Casi todos los tutoriales de ML asumen que ya tienes un `y` listo:
spam/no-spam, enfermo/sano, etc. En la vida real, muchas veces **tú
tienes que decidir qué define el `y`**. Esta demo muestra el ejercicio
sobre un dataset sintético de usuarios de e-commerce donde queremos
predecir *"¿a quién vale la pena enviarle una promoción personalizada?"*.

El dataset se genera con `generador_datos.py` (10 000 usuarios con
valores faltantes realistas) y el target `dar_promocion` se construye a
partir de reglas sobre `last_purchase_days`, `avg_order_value`,
`cart_abandonment_rate` y otras variables.

## 🎯 Objetivo de aprendizaje

Al terminar esta demo deberías poder responder:

- ¿Qué pasa cuando el dataset NO trae el target definido?
- ¿Cómo se operacionaliza una pregunta de negocio en una regla concreta?
- ¿Cómo se construye un pipeline `ColumnTransformer` con numéricas y categóricas?
- ¿Cómo se interpreta `classification_report` y la matriz de confusión?
- ¿Qué cambia al mover el umbral de clasificación?

## 📂 Qué hay en la carpeta

| Archivo | Qué trae |
|---|---|
| `01_mlops_intro_notebook.ipynb` | El notebook completo: generación de datos → EDA → feature engineering → split → pipeline → métricas → umbral |
| `generador_datos.py` | Clase `UserGenerator` que crea 10 000 usuarios sintéticos y marca `dar_promocion` según reglas de negocio |

## 🚀 Cómo usar esta demo

### Paso 1 — Instala las dependencias

Las de esta demo ya vienen con la base del repo (pandas, numpy, sklearn,
matplotlib, seaborn). Desde la **raíz del repo**:

```bash
uv sync --extra dev
```

### Paso 2 — Abre el notebook

```bash
cd projects/d3-promociones-usuarios
uv run jupyter lab 01_mlops_intro_notebook.ipynb
```

### Paso 3 — Recorre las 7 secciones (en orden)

El notebook está estructurado como un recorrido corto:

| # | Sección | Qué hace |
|---|---|---|
| 1 | EDA rápido | Mira tipos, nulos, distribuciones. Antes de modelar |
| 2 | Feature engineering | Crea `total_purchases_per_day`, `bucket_avg_order_value`, etc. |
| 3 | Split train/test | Separación aleatoria con `random_state=42` |
| 4 | Preprocesamiento | Numéricas vs. categóricas, por qué cada una |
| 5 | Pipeline + Logistic Regression | Un solo objeto reproducible; **sin fuga** |
| 6 | Métricas de clasificación | `classification_report`, matriz de confusión |
| 7 | Umbral y trade-off precision/recall | Pruebas con umbral 0.3, 0.5, 0.7 |

Al final hay una **actividad opcional** con 7 variaciones para que las
pruebes tú.

## 🎓 Qué debes aprender en 20 minutos

1. El `y` no siempre viene dado. Alguien **lo construyó** con una regla.
2. Esa regla es una decisión de negocio, no técnica. Se discute con marketing, no con el modelo.
3. Un `Pipeline` + `ColumnTransformer` evita fuga de datos en una línea.
4. Accuracy sola engaña con clases desbalanceadas. Usa precision/recall/F1 y la matriz.
5. El umbral por defecto (0.5) es una convención, no una ley. Se mueve según el costo de equivocarse.

## 📝 Notas pedagógicas

- El dataset es **sintético a propósito**. Las reglas que generan
  `dar_promocion` son conocidas; el modelo debería recuperar al menos
  algunas. No esperes AUC alta: el ruido que agregó el generador la
  tiene cerca de 0.5 — lo cual es un hallazgo pedagógico útil
  ("tus features no están explicando el target que tú mismo diseñaste").
- El ROC-AUC bajo del Logistic Regression (~0.52) es intencional: el
  objetivo de la demo es el flujo, no el score. Probar `RandomForest` o
  más features (parte de la actividad final) sube el número.
- Si quieres profundizar en umbrales como tema aparte, mira la demo
  [`d4-umbrales-ejemplo/`](../d4-umbrales-ejemplo/). Si quieres
  profundizar en pipelines, mira
  [`d1-ejercicio-preprocesamiento/`](../d1-ejercicio-preprocesamiento/).

## 📝 Actividad para después

El notebook cierra con una lista de 7 variaciones para probar por tu
cuenta:

1. Imputar las numéricas con otras estrategias.
2. Crear nuevas features.
3. Usar `BinaryEncoder` para las categóricas.
4. Usar `MinMaxScaler` para las numéricas.
5. Usar `RandomForestClassifier` como modelo.
6. Usar `GridSearchCV` para hiperparámetros.
7. Jugar con el split de datos.

Entrega: PR al repo con tu solución.

## 🔗 Relación con otros materiales

- Si no tienes claro qué es *data leakage*, pasa por
  [`d1-ejercicio-preprocesamiento/`](../d1-ejercicio-preprocesamiento/)
  antes.
- Para split con componente temporal, mira
  [`d2-temporal-split/`](../d2-temporal-split/).
- Para umbrales en más detalle, mira
  [`d4-umbrales-ejemplo/`](../d4-umbrales-ejemplo/).
- Para una referencia completa del flujo (notebook → módulo → app),
  mira [`05-ejemplo-completo/`](../05-ejemplo-completo/).

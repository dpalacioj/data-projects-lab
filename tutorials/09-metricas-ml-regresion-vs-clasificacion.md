---
title: "09 · Métricas regresión vs. clasificación"
parent: Tutoriales
nav_order: 11
---
# Tutorial 05: Métricas de regresión y clasificación

> Reescrito desde **fuentes primarias**:
> - Hastie, Tibshirani, Friedman (2009). *The Elements of Statistical Learning*, 2nd ed. (capítulos 2, 7 y 9).
> - Scikit-learn user guide, sección [3.4 Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).
> - Davis, Goadrich (2006). *The Relationship Between Precision-Recall and ROC Curves*. ICML.
>
> Todas las figuras son **propias**, generadas desde el caso de recompra
> del proyecto `segmentacion-clientes` (`scripts/figuras_u3.py`).

## Índice

1. [Por qué medimos](#por-qué-medimos)
2. [Métricas de regresión](#métricas-de-regresión)
3. [Métricas de clasificación](#métricas-de-clasificación)
4. [Precisión vs. Recall: la decisión del umbral](#precisión-vs-recall-la-decisión-del-umbral)
5. [AUC y Precisión Media](#auc-y-precisión-media)
6. [Sobreajuste y la curva de validación](#sobreajuste-y-la-curva-de-validación)
7. [Qué métrica usar y cuándo](#qué-métrica-usar-y-cuándo)
8. [Caso aplicado: recompra en 90 días](#caso-aplicado-recompra-en-90-días)

---

## Por qué medimos

Un modelo de ML produce predicciones. **Las métricas son el puente entre
las predicciones y la decisión de negocio.** Antes de escribir código,
hay que acordar con quien va a actuar sobre esas predicciones:

- ¿Qué cuesta un **falso positivo**? ¿Y un **falso negativo**?
- ¿Buscamos predecir **un valor numérico** (regresión) o una **clase**
  (clasificación)?
- ¿Queremos predecir **sí/no** o **una probabilidad**?

Un modelo con "accuracy 95 %" sin esa conversación de negocio detrás es
un número aislado. Las métricas son útiles cuando se vinculan a una
decisión concreta.

## Métricas de regresión

Dado un vector real `y` y una predicción `ŷ`:

| Métrica | Fórmula | Interpretación | Robustez a outliers |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | `mean(|y − ŷ|)` | Error medio en las unidades originales. | Alta — penaliza linealmente. |
| **MSE** (Mean Squared Error) | `mean((y − ŷ)²)` | Error cuadrático; penaliza desviaciones grandes. | Baja — los outliers dominan. |
| **RMSE** | `√MSE` | Mismas unidades que `y`. | Baja — hereda la de MSE. |
| **R²** (coeficiente de determinación) | `1 − SS_res / SS_tot` | Fracción de la varianza explicada; **negativo** si el modelo es peor que la media. | Depende de la escala de `y`. |

**Hastie et al. (2009), §7.2** justifican el MSE como la función de
pérdida natural bajo el supuesto de ruido gaussiano. En producción, el
MAE suele ser más interpretable para el negocio ("el modelo se
equivoca en promedio ±X unidades"), mientras que el RMSE pesa más los
errores grandes.

### Cuándo elegir cada una

- **MAE**: cuando los errores tienen el mismo costo en todas las
  magnitudes (p.ej. pronóstico de ventas en unidades físicas).
- **RMSE / MSE**: cuando los errores grandes son desproporcionadamente
  costosos (p.ej. estimación de precios en inmuebles).
- **R²**: para comparar modelos entre sí sobre el **mismo** dataset.
  No compara datasets distintos.

## Métricas de clasificación

Para clasificación binaria, todo nace de la **matriz de confusión**:

|  | Predicho positivo | Predicho negativo |
|---|---|---|
| **Positivo real** | TP (verdadero positivo) | FN (falso negativo) |
| **Negativo real** | FP (falso positivo) | TN (verdadero negativo) |

Las métricas derivadas:

| Métrica | Fórmula | Lectura |
|---|---|---|
| **Accuracy** | `(TP + TN) / total` | ¿Qué fracción está bien? Engaña con clases desbalanceadas. |
| **Precision** | `TP / (TP + FP)` | De los que señaló el modelo, ¿cuántos acertó? |
| **Recall / Sensibilidad** | `TP / (TP + FN)` | De los positivos reales, ¿cuántos encontró? |
| **F1** | `2·P·R / (P + R)` | Media armónica de P y R; penaliza desequilibrios. |
| **Especificidad** | `TN / (TN + FP)` | De los negativos reales, ¿cuántos encontró? |

### El riesgo de la *accuracy* en datasets desbalanceados

Si el 95 % de los casos son negativos, un modelo que predice **siempre
negativo** tiene accuracy 95 %. Es **inútil** para detectar positivos.
En U3 del módulo (caso recompra), la clase mayoritaria en la cohorte B
tiene accuracy **0,573**: ese es el "mínimo honesto" que cualquier
modelo debe superar.

## Precisión vs. Recall: la decisión del umbral

Un clasificador probabilístico produce `p ∈ [0, 1]`. La decisión
"positivo si `p ≥ umbral`" convierte probabilidades en decisiones. El
umbral **no** es una propiedad del modelo: es una decisión de negocio.

- **Umbral alto (p.ej. 0,7)**: pocas predicciones positivas, pero muy
  seguras. **Alta precisión**, bajo recall.
- **Umbral bajo (p.ej. 0,2)**: muchas predicciones positivas; se cae
  más veces, pero no se escapa casi ningún positivo real. **Alto
  recall**, baja precisión.

Es un trade-off puro: cualquier mejora en precisión suele venir con una
caída en recall, y viceversa.

### Analogía

Imagina un radar de pesca:

- **Alta precisión + bajo recall**: solo lanza la red cuando está muy
  seguro de que hay peces. Pierde algunas oportunidades, pero no gasta
  esfuerzo en vano.
- **Alto recall + baja precisión**: lanza la red cada vez que
  "sospecha". Captura más peces en total, pero muchas veces no pesca
  nada.

Depende del costo del esfuerzo (precisión alta) vs. el costo de perder
peces (recall alto).

## AUC y Precisión Media

Dos métricas que **integran sobre todos los umbrales**:

- **AUC (Area Under the ROC Curve)**: probabilidad de que el modelo
  dé una probabilidad más alta a un positivo que a un negativo elegido
  al azar. Independiente del umbral. 0,5 = azar; 1 = perfecto.
- **AP (Average Precision)**: área bajo la curva precisión-recall. Más
  informativa que AUC cuando la clase positiva es minoritaria (Davis
  & Goadrich, 2006).

> La figura `g_roc_pr.png` del proyecto `segmentacion-clientes`
> (generada por `scripts/figuras_u3.py`) muestra ambas curvas con las
> marcas de umbral `0,3` y `0,5`.

![ROC y PR](../projects/00-segmentacion-clientes/reports/figures/g_roc_pr.png)

### Qué métrica elegir

- **Clases balanceadas**: AUC es apropiada.
- **Clase positiva rara**: AP mide mejor la utilidad de un ranking. Un
  modelo con AUC alta y AP baja todavía ranquea mejor el ruido que lo
  interesante.

## Sobreajuste y la curva de validación

Un modelo que memoriza el conjunto de entrenamiento parece excelente
sobre esos datos, pero falla al generalizar. **Hastie et al. (2009),
§7.4** describen la relación entre complejidad del modelo y error de
test como una curva en forma de U.

> La figura `g_sobreajuste.png` lo muestra sobre el caso de recompra:
> AUC(A) crece monótonamente con la profundidad del árbol (de 0,696 a
> 1,000); AUC(B) crece al principio y **cae** después de profundidad 4.

![Sobreajuste](../projects/00-segmentacion-clientes/reports/figures/g_sobreajuste.png)

### Cómo detectarlo

- Comparar AUC/accuracy del **train** vs. **test**: si la diferencia
  crece con la complejidad, el modelo está memorizando.
- Usar **validación cruzada**: si CV AUC ≈ test AUC, no hay fuga
  temporal ni sobreajuste severo.
- Fuga de datos: el modelo que incluye `recencia_fuga` (variable que
  usa información posterior al corte) salta de AUC(B) = 0,795 a 0,955.
  Un salto así de grande sin cambio de arquitectura es casi siempre
  fuga, no mejora real.

![Fuga](../projects/00-segmentacion-clientes/reports/figures/g_fuga.png)

## Qué métrica usar y cuándo

| Problema | Métrica principal | Métricas secundarias |
|---|---|---|
| Regresión balanceada | **RMSE** | MAE, R² |
| Regresión con outliers | **MAE** | RMSE, quantile loss |
| Clasificación balanceada | **AUC** | Accuracy, F1 |
| Clasificación con clase rara | **AP** | Recall @ umbral fijo, F1 |
| Operación crítica con costo asimétrico | **Precisión** o **Recall** según costo | F1, tabla de umbrales |

### Antes de elegir, hazte estas tres preguntas

1. **¿Qué cuesta cada tipo de error?** FP y FN rara vez cuestan lo mismo.
2. **¿Qué le vas a reportar al negocio?** Si no entienden la métrica,
   no la van a usar para decidir. MAE suele ser más intuitiva que RMSE.
3. **¿Cuál es el baseline?** El modelo debe superar un baseline simple
   (clase mayoritaria, regla de dominio) **y** por un margen que
   justifique su mantenimiento.

## Caso aplicado: recompra en 90 días

El proyecto `segmentacion-clientes` tiene un caso supervisado completo
donde todas estas métricas se calculan de verdad:

- **Pregunta de negocio**: ¿este cliente volverá a comprar en los
  próximos 90 días?
- **Datos**: Online Retail II (UCI 502), 2 años de transacciones.
- **Modelos**: regresión logística, árbol de profundidad 4, bosque de
  300 árboles.
- **Línea base**: regla "quien compró hace menos de 90 días comprará
  otra vez". Accuracy 0,706, precisión 0,681, recall 0,585, F1 0,629.
- **Mejor modelo**: regresión logística, AUC(B) = 0,795. Supera a la
  regla en AUC, no por mucho en F1 — es el equilibrio honesto.

Flujo completo en los cuadernos:

- [`06_particion_y_fuga.ipynb`](../projects/00-segmentacion-clientes/notebooks/06_particion_y_fuga.ipynb)
  — cómo construir cohortes sin fuga.
- [`09_modelos_y_pipelines.ipynb`](../projects/00-segmentacion-clientes/notebooks/09_modelos_y_pipelines.ipynb)
  — las métricas aplicadas con pipelines reproducibles y curva de
  profundidad; importancias del bosque:

![Importancia](../projects/00-segmentacion-clientes/reports/figures/g_importancia.png)

---

## Para ir más allá

- **Hastie et al. (2009)** §7.10 — selección de modelos y validación
  cruzada.
- **Scikit-learn metrics guide** — glosario completo de métricas con
  ejemplos ejecutables.
- **Davis & Goadrich (2006)** — por qué PR puede ser más informativa
  que ROC para clases desbalanceadas.

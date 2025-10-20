# Tutorial: Métricas de Evaluación y Probabilidades en Algoritmos de Clasificación

## Introducción

Cuando entrenamos un modelo de machine learning, necesitamos entender **qué tan bien funciona**. Este tutorial explica las métricas más importantes para evaluar modelos de clasificación y cómo funcionan las probabilidades de predicción.

---

## 1. ¿Cómo Predicen los Algoritmos?

### 1.1 Probabilidades vs Etiquetas

La mayoría de los algoritmos de clasificación no predicen directamente "Sí" o "No". En lugar de eso:

1. **Calculan probabilidades**: El modelo estima la probabilidad de que una muestra pertenezca a cada clase
   - Ejemplo: "Esta persona tiene 75% de probabilidad de sobrevivir"

2. **Aplican un umbral**: Se usa un valor de corte (por defecto 0.5) para convertir probabilidades en predicciones finales

```
Probabilidad >= 0.5  →  Predicción = 1 (Clase positiva)
Probabilidad < 0.5   →  Predicción = 0 (Clase negativa)
```

### 1.2 El Umbral de Decisión

**Definición**: El umbral (threshold) es el valor que separa las predicciones positivas de las negativas.

**Ejemplo práctico - Predicción de Titanic:**

```
Pasajero A: Probabilidad de sobrevivir = 0.85  →  Predicción = Sobrevive ✓
Pasajero B: Probabilidad de sobrevivir = 0.62  →  Predicción = Sobrevive ✓
Pasajero C: Probabilidad de sobrevivir = 0.48  →  Predicción = No sobrevive ✗
Pasajero D: Probabilidad de sobrevivir = 0.12  →  Predicción = No sobrevive ✗
```

**¿Por qué es importante?**
- Podemos **ajustar el umbral** según nuestras necesidades
- Umbral más bajo (ej: 0.3) → Más predicciones positivas (mayor sensibilidad)
- Umbral más alto (ej: 0.7) → Menos predicciones positivas (mayor especificidad)

**Caso de uso real:**
- **Diagnóstico médico**: Umbral bajo (0.3) para no perder casos positivos
- **Detección de spam**: Umbral alto (0.7) para evitar marcar correos importantes como spam

---

## 2. La Matriz de Confusión

Antes de entender las métricas, necesitamos conocer la **matriz de confusión**:

```
                    Predicción
                 Negativo  Positivo
Real  Negativo      TN        FP
      Positivo      FN        TP
```

**Componentes:**
- **TP (True Positives)**: Predijimos positivo y era positivo ✅
- **TN (True Negatives)**: Predijimos negativo y era negativo ✅
- **FP (False Positives)**: Predijimos positivo pero era negativo ❌ (Error Tipo I)
- **FN (False Negatives)**: Predijimos negativo pero era positivo ❌ (Error Tipo II)

**Ejemplo - Titanic:**
```
Predicción: ¿Sobrevive?
                      NO    SÍ
Real    NO           147    15    (162 no sobrevivieron)
        SÍ            29    71    (100 sobrevivieron)
```

---

## 3. Métricas de Evaluación

### 3.1 Accuracy (Exactitud)

**Definición**: Proporción de predicciones correctas sobre el total.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretación**: "¿Cuántas predicciones hice bien?"

**Ejemplo:**
```
Accuracy = (147 + 71) / (147 + 71 + 15 + 29) = 218/262 = 0.832 → 83.2%
```

**Ventajas:**
- Fácil de entender
- Métrica intuitiva

**Limitaciones:**
- **No funciona bien con datos desbalanceados**
  - Ejemplo: Si 95% de los datos son negativos, un modelo que siempre predice negativo tendrá 95% de accuracy pero es inútil

**¿Cuándo usarla?**
- Cuando las clases están **balanceadas** (similar cantidad de positivos y negativos)

---

### 3.2 Precision (Precisión)

**Definición**: De todas las predicciones positivas que hice, ¿cuántas eran correctas?

```
Precision = TP / (TP + FP)
```

**Interpretación**: "Cuando digo que SÍ, ¿qué tan confiable soy?"

**Ejemplo:**
```
Precision = 71 / (71 + 15) = 71/86 = 0.826 → 82.6%
```

**Significado**: De cada 100 personas que predije que sobrevivirían, 83 realmente sobrevivieron.

**¿Cuándo es importante?**
- **Detección de spam**: No queremos marcar correos importantes como spam (evitar FP)
- **Recomendaciones de productos**: No queremos recomendar productos irrelevantes

---

### 3.3 Recall (Sensibilidad o Exhaustividad)

**Definición**: De todos los casos positivos reales, ¿cuántos detecté?

```
Recall = TP / (TP + FN)
```

**Interpretación**: "De todos los casos positivos que existen, ¿cuántos logré encontrar?"

**Ejemplo:**
```
Recall = 71 / (71 + 29) = 71/100 = 0.71 → 71%
```

**Significado**: De cada 100 personas que sobrevivieron, detecté correctamente a 71.

**¿Cuándo es importante?**
- **Diagnóstico de enfermedades**: No queremos perder ningún caso positivo (evitar FN)
- **Detección de fraude**: Queremos capturar todos los fraudes posibles
- **Sistemas de seguridad**: Mejor tener falsas alarmas que dejar pasar una amenaza

---

### 3.4 El Trade-off Precision vs Recall

**Concepto clave**: Generalmente, mejorar una métrica empeora la otra.

**Ejemplo con diferentes umbrales:**

| Umbral | Precision | Recall | Explicación |
|--------|-----------|--------|-------------|
| 0.9    | 95%       | 40%    | Solo predicciones muy seguras → Pocas pero correctas |
| 0.5    | 83%       | 71%    | Balance estándar |
| 0.2    | 60%       | 95%    | Muchas predicciones → Capturamos casi todo pero con errores |

**¿Qué priorizar?**
- **Alta Precision**: Cuando los falsos positivos son costosos (spam, recomendaciones)
- **Alto Recall**: Cuando los falsos negativos son costosos (enfermedades, fraude)

---

### 3.5 AUC-ROC (Area Under the Curve)

**Definición**: El AUC mide la capacidad del modelo para **distinguir entre clases** en todos los umbrales posibles.

**ROC Curve**: Gráfico que muestra la relación entre:
- **Eje Y**: True Positive Rate (Recall)
- **Eje X**: False Positive Rate (FP / (FP + TN))

**Interpretación del AUC:**
```
AUC = 1.0    →  Modelo perfecto (100% de discriminación)
AUC = 0.9    →  Modelo excelente
AUC = 0.8    →  Modelo bueno
AUC = 0.7    →  Modelo aceptable
AUC = 0.5    →  Modelo aleatorio (inútil, como lanzar una moneda)
AUC < 0.5    →  Modelo peor que aleatorio
```

**Significado intuitivo**:
Si tomo un caso positivo y uno negativo al azar, el AUC es la probabilidad de que el modelo asigne mayor probabilidad al caso positivo.

**Ventajas:**
- **Independiente del umbral**: Evalúa el modelo en general
- **Funciona bien con datos desbalanceados**
- Métrica robusta para comparar modelos

**¿Cuándo usarla?**
- Para **comparar modelos** sin preocuparse por el umbral específico
- Cuando necesitas una métrica global de desempeño

---

### 3.6 Kappa de Cohen

**Definición**: Mide el acuerdo entre las predicciones y los valores reales, **corrigiendo por acuerdo aleatorio**.

```
Kappa = (Accuracy observada - Accuracy esperada por azar) / (1 - Accuracy esperada por azar)
```

**Interpretación:**

| Valor Kappa | Interpretación |
|-------------|----------------|
| < 0.0       | Peor que azar |
| 0.0 - 0.20  | Acuerdo leve |
| 0.21 - 0.40 | Acuerdo justo |
| 0.41 - 0.60 | Acuerdo moderado |
| 0.61 - 0.80 | Acuerdo sustancial |
| 0.81 - 1.00 | Acuerdo casi perfecto |

**¿Por qué es útil?**
- Corrige el efecto de **clases desbalanceadas**
- Si 90% de los datos son negativos, predecir siempre "negativo" da 90% accuracy pero Kappa cercano a 0

**Ejemplo:**
```
Accuracy = 83%
Kappa = 0.63  →  Acuerdo sustancial (el modelo aprende patrones reales, no azar)
```

**¿Cuándo usarla?**
- Cuando tienes **datos desbalanceados**
- Para validar que el modelo no está simplemente adivinando la clase mayoritaria

---

### 3.7 MCC (Matthews Correlation Coefficient)

**Definición**: Coeficiente de correlación entre predicciones y valores reales. Es considerada una de las **mejores métricas únicas** para clasificación binaria.

```
MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Interpretación:**

| Valor MCC | Interpretación |
|-----------|----------------|
| +1.0      | Predicción perfecta |
| 0.0       | Predicción aleatoria |
| -1.0      | Desacuerdo total (predicciones inversas) |

**Ventajas:**
- **Balanceada**: Considera los 4 elementos de la matriz de confusión por igual
- **Robusta con datos desbalanceados**
- Métrica simétrica (trata FP y FN con igual importancia)

**Comparación con otras métricas:**
```
Escenario: 5 positivos, 95 negativos
Modelo predice todo negativo:
- Accuracy: 95% (engañoso)
- Kappa: 0 (neutral)
- MCC: 0 (neutral)

Modelo balanceado:
- Accuracy: 85%
- Kappa: 0.52
- MCC: 0.55
```

**¿Cuándo usarla?**
- Cuando necesitas **una sola métrica confiable**
- Con **datos desbalanceados**
- Para **comparar modelos de forma justa**

---

## 4. ¿Qué Métrica Elegir?

### Guía de Decisión:

```
┌─────────────────────────────────────────────────────────┐
│ ¿Tus clases están balanceadas (50/50 aprox)?          │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │ SÍ                │ NO
        │                   │
        v                   v
    Accuracy          MCC o Kappa
    Precision         (corrigen desbalance)
    Recall
    F1-Score
                            │
                ┌───────────┴───────────┐
                │ ¿Qué es más costoso? │
                └───────────┬───────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            v                               v
    Falsos Positivos              Falsos Negativos
    (FP son peores)               (FN son peores)
            │                               │
            v                               v
      Precision                        Recall
      (ej: spam)                    (ej: cáncer)
```

### Resumen por Caso de Uso:

| Caso de Uso | Métrica Principal | Métrica Secundaria | Razón |
|-------------|-------------------|-------------------|-------|
| **Diagnóstico médico** | Recall | AUC | No queremos perder casos positivos |
| **Spam** | Precision | F1-Score | No queremos marcar correos importantes |
| **Fraude** | Recall, AUC | MCC | Capturar todos los fraudes posibles |
| **Clasificación balanceada** | Accuracy | AUC | Clases equilibradas |
| **Clasificación desbalanceada** | MCC | Kappa, AUC | Corrige el desbalance |
| **Comparación de modelos** | AUC | MCC | Métrica independiente del umbral |

---

## 5. Ejemplo Práctico Completo

### Escenario: Predicción de Supervivencia en Titanic

**Matriz de Confusión:**
```
                    Predicción
                 No Sobrevive  Sobrevive
Real  No Sobrevive    147         15     (162)
      Sobrevive        29         71     (100)
```

**Cálculo de todas las métricas:**

```python
TP = 71  (correctamente predicho sobrevive)
TN = 147 (correctamente predicho no sobrevive)
FP = 15  (predije sobrevive pero no sobrevivió)
FN = 29  (predije no sobrevive pero sobrevivió)

# Accuracy
Accuracy = (71 + 147) / 262 = 0.832 → 83.2%

# Precision
Precision = 71 / (71 + 15) = 0.826 → 82.6%

# Recall
Recall = 71 / (71 + 29) = 0.710 → 71.0%

# MCC
MCC = (71×147 - 15×29) / √[(71+15)(71+29)(147+15)(147+29)]
    = (10437 - 435) / √[86×100×162×176]
    = 10002 / 17,235
    = 0.580 → Correlación moderada-fuerte
```

**Interpretación:**
- ✅ **Accuracy 83%**: Modelo preciso en general
- ✅ **Precision 83%**: Cuando predecimos "sobrevive", acertamos 8 de cada 10 veces
- ⚠️ **Recall 71%**: Nos perdimos el 29% de los sobrevivientes (29 personas)
- ✅ **MCC 0.58**: Modelo aprende patrones reales (no es azar)

**Decisión**: Si el objetivo es **salvar vidas**, deberíamos aumentar el Recall (bajar umbral) aunque perdamos algo de Precision.

---

## 6. Ejercicio Práctico

**Situación**: Tienes un modelo de detección de cáncer con estos resultados:

```
                    Predicción
                 Sano    Enfermo
Real  Sano       950       50     (1000)
      Enfermo     10       90     (100)
```

**Calcula:**
1. Accuracy
2. Precision
3. Recall
4. ¿Es un buen modelo? ¿Por qué?
5. ¿Qué métrica es más importante aquí?

<details>
<summary>Ver soluciones</summary>

```
1. Accuracy = (950 + 90) / 1100 = 0.945 → 94.5%

2. Precision = 90 / (90 + 50) = 0.643 → 64.3%

3. Recall = 90 / (90 + 10) = 0.900 → 90%

4. Es un modelo DECENTE pero no excelente porque:
   - Accuracy alta (94.5%) pero engañosa (datos desbalanceados)
   - Precision baja (64.3%): Muchos falsos positivos (alarmas innecesarias)
   - Recall alto (90%): Solo perdemos 10 casos de cáncer (aceptable)

5. En medicina, RECALL es crítico: preferimos 50 falsos positivos
   (personas sanas que trataremos por precaución) que perder 10 casos
   de cáncer. La vida humana es prioritaria.
```

</details>

---

## 7. Conclusiones

1. **Las probabilidades** son el resultado "crudo" del modelo; el **umbral** las convierte en decisiones

2. **No existe una métrica perfecta**: depende del problema y los costos de los errores

3. **Datos desbalanceados**: Usa MCC, Kappa o AUC en lugar de Accuracy

4. **Precision vs Recall**: Es un balance. Decide según el costo de cada tipo de error

5. **AUC**: Mejor métrica para comparar modelos sin importar el umbral

6. **MCC**: Métrica más robusta y balanceada para clasificación binaria

---

## Recursos Adicionales

- **Scikit-learn**: Documentación sobre métricas de clasificación
- **ROC Curves**: Visualización interactiva en scikit-learn
- **Confusion Matrix**: Herramientas de visualización con seaborn

---

**Licencia**: MIT License
**Autor**: David Palacio Jiménez
**Fecha**: 2025

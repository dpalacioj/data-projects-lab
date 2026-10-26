# Umbrales de Clasificación

## 📚 Descripción

Este proyecto ilustra cómo el ajuste del umbral de clasificación (threshold) afecta las predicciones y métricas de desempeño en modelos de clasificación binaria, especialmente en escenarios con clases desbalanceadas.

## 🎯 Objetivo de Aprendizaje

Entender que:
- Los modelos de clasificación predicen **probabilidades**, no clases directamente
- El umbral convierte probabilidades en decisiones binarias
- El umbral por defecto (0.5) no siempre es óptimo
- Existe un trade-off entre Precision y Recall controlado por el umbral

## 📖 Teoría: Umbral de Clasificación

### ¿Qué es el umbral?

En clasificación binaria, los modelos de machine learning generan una **probabilidad** P(y=1) para cada observación. Para convertir esta probabilidad en una predicción de clase, aplicamos un **umbral** (threshold):

```
Si P(y=1) >= umbral → Predecir clase 1 (positivo)
Si P(y=1) < umbral  → Predecir clase 0 (negativo)
```

**Por defecto, el umbral es 0.5**, pero esto es arbitrario y no siempre óptimo.

### ¿Por qué ajustar el umbral?

1. **Clases desbalanceadas**: Cuando una clase es minoritaria (ej: 5% de casos positivos), el umbral 0.5 puede ser inadecuado.

2. **Costos asimétricos**: En muchos problemas, los errores tienen costos diferentes:
   - Falso Negativo en diagnóstico médico → Muy costoso
   - Falso Positivo en spam filtering → Menos costoso

3. **Objetivos de negocio**: Diferentes aplicaciones requieren diferentes trade-offs.

### El Trade-off Precision-Recall

Al ajustar el umbral, modificamos el balance entre Precision y Recall:

| Umbral | Predicciones Positivas | Precision | Recall | Uso típico |
|--------|------------------------|-----------|--------|------------|
| **Bajo (ej: 0.2)** | Más predicciones como positivas | ↓ Baja | ↑ Alto | Detección de fraude, enfermedades |
| **Medio (0.5)** | Balance estándar | Media | Medio | Default general |
| **Alto (ej: 0.8)** | Menos predicciones como positivas | ↑ Alta | ↓ Bajo | Marketing dirigido, precisión crítica |

#### Precision (Precisión)

$$\text{Precision} = \frac{TP}{TP + FP}$$

De todas las predicciones positivas, ¿cuántas son correctas?

#### Recall (Sensibilidad, Cobertura)

$$\text{Recall} = \frac{TP}{TP + FN}$$

De todos los casos positivos reales, ¿cuántos capturamos?

### Métodos para Seleccionar el Umbral Óptimo

1. **Criterio de Youden (J-statistic)**
   - Maximiza: J = Sensibilidad + Especificidad - 1
   - Equilibra TPR y FPR
   - Usado en este notebook

2. **Maximizar F1-Score**
   - Balance armónico entre Precision y Recall
   - Útil cuando ambas métricas son igualmente importantes

3. **Curva Precision-Recall**
   - Analizar el punto de mejor balance según necesidades del negocio

4. **Costo-beneficio**
   - Asignar costos a FP y FN
   - Minimizar el costo total esperado

## 🚀 Cómo usar este proyecto

### Paso 1 — Instala las dependencias

Esta demo usa LightGBM, que no está en la base. Desde la **raíz del repo**:

```bash
uv add lightgbm --dev           # añade LightGBM al grupo dev del repo
uv sync --extra dev
```

(Alternativa simple: `uv run --with lightgbm jupyter lab umbral_clasificacion.ipynb`
desde esta carpeta, sin modificar `pyproject.toml`.)

### Paso 2 — Abre el notebook

```bash
cd projects/d4-umbrales-ejemplo
uv run jupyter lab umbral_clasificacion.ipynb
```

### Paso 3 — Qué hacer en clase

Ejecuta las celdas secuencialmente. Observa cómo cambia la relación
precision/recall al mover el umbral de 0.1 a 0.9.

**Objetivo pedagógico (20 min):** romper el reflejo de usar siempre 0.5
y entender que el umbral se elige con el equipo de negocio según el
costo de falsos positivos vs. falsos negativos.

## 📊 Contenido del Notebook

1. **Generación de datos sintéticos** con clases desbalanceadas (90%-10%)
2. **Entrenamiento de LightGBM** para obtener probabilidades
3. **Evaluación con umbral por defecto (0.5)**
4. **Experimentación con diferentes umbrales** (0.1 a 0.9)
5. **Análisis de la curva ROC** y selección de umbral óptimo
6. **Comparación de resultados** entre umbrales

## 🎓 Casos de Uso Reales

### Umbral Bajo (Maximizar Recall)
- **Diagnóstico de enfermedades**: Preferimos capturar todos los casos positivos (pocos FN)
- **Detección de fraude**: No queremos perder transacciones fraudulentas
- **Sistemas de alerta**: Mejor prevenir que lamentar

### Umbral Alto (Maximizar Precision)
- **Marketing dirigido**: Solo contactar clientes con alta probabilidad de conversión
- **Recomendaciones críticas**: Evitar sugerencias irrelevantes
- **Filtrado de contenido**: Reducir falsos positivos molestos

## 📝 Conclusiones Clave

1. El umbral de 0.5 es una convención, **no una regla universal**
2. Clases desbalanceadas requieren ajuste de umbral casi siempre
3. La elección del umbral depende del **contexto y objetivos del problema**
4. Siempre analiza Precision, Recall y la curva ROC antes de decidir
5. El mejor umbral es el que **minimiza el costo real del error** en tu aplicación

## 📚 Referencias

- [Scikit-learn: Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Google ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
- Provost, F., & Fawcett, T. (2001). "Robust Classification for Imprecise Environments"

## 👨‍🏫 Para Instructores

Este material es ideal para:
- Cursos de Machine Learning nivel intermedio
- Módulos sobre métricas de clasificación
- Talleres sobre clases desbalanceadas
- Proyectos prácticos con datos reales desbalanceados

**Duración estimada**: 45-60 minutos

---

**Licencia**: MIT License
**Autor**: David Palacio Jiménez

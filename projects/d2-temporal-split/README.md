# Data Leakage Temporal: Train/Test Split Correcto

## 📚 Descripción

Este proyecto ilustra uno de los errores más comunes y peligrosos en machine learning aplicado a datos temporales: **usar train/test split aleatorio cuando los datos tienen una dimensión temporal**.

## 🎯 Objetivo de Aprendizaje

Comprender que:
- Los datos con componente temporal requieren un split especial
- El split aleatorio tradicional causa **data leakage temporal**
- Las métricas pueden ser artificialmente optimistas si no se respeta el orden temporal
- El desempeño real en producción puede ser mucho peor de lo esperado

## 📖 Teoría: Data Leakage Temporal

### ¿Qué es Data Leakage?

**Data leakage** (fuga de datos) ocurre cuando información del conjunto de test "se filtra" al conjunto de entrenamiento, permitiendo que el modelo aprenda patrones que no estarían disponibles en un escenario real de predicción.

### El Problema Específico con Datos Temporales

En muchos problemas de machine learning, los datos tienen una **dimensión temporal**:
- Transacciones financieras con timestamps
- Ventas diarias/mensuales
- Logs de sistemas
- Métricas de sensores
- Series de tiempo en general

**¿Qué ocurre con un split aleatorio?**

```python
# ❌ INCORRECTO
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True  # shuffle=True es el problema
)
```

Al mezclar aleatoriamente, podemos tener:
- Datos de **diciembre 2023** en el conjunto de entrenamiento
- Datos de **enero 2023** en el conjunto de test

**Consecuencia**: El modelo aprende del "futuro" (diciembre) para predecir el "pasado" (enero).

### ¿Por qué es tan peligroso?

1. **Métricas infladas**: El modelo parece funcionar mejor de lo que realmente funcionará en producción
2. **Falsa confianza**: Tomamos decisiones basadas en métricas que no reflejan la realidad
3. **Fallo en producción**: Al desplegar, el modelo solo conoce el pasado, no el futuro
4. **Impacto en negocio**: Predicciones incorrectas pueden costar dinero real

### Ejemplo Concreto

Imagina que predices ventas de un producto:

**Con split aleatorio (incorrecto):**
- Tu modelo aprende que las ventas en diciembre son altas (temporada navideña)
- Cuando predice ventas de octubre (que está en test), usa información de diciembre
- Métricas: R² = 0.95 (¡excelente!)

**En producción:**
- En octubre, el modelo NO conoce diciembre todavía
- Las predicciones son malas porque el modelo dependía de información futura
- Desempeño real: R² = 0.65 (mediocre)

## 🔧 La Solución: Split Temporal

### Principio Fundamental

> **El conjunto de entrenamiento debe contener SOLO datos anteriores al conjunto de test**

Esto simula el escenario real donde:
- Aprendemos del **pasado** (train)
- Predecimos el **futuro** (test)

### Implementación Correcta

```python
# ✅ CORRECTO: Split temporal manual
split_point = int(len(df) * 0.7)

X_train = X.iloc[:split_point]      # Primeros 70% (datos antiguos)
X_test = X.iloc[split_point:]       # Últimos 30% (datos recientes)
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]
```

### Visualización del Concepto

```
Línea de tiempo: ───────────────────────────▶
                 2022-01-01          2023-12-31

Split Aleatorio (INCORRECTO):
Train:  ██░░██░░░██░░██████░░░██
Test:   ░░██░░███░░██░░░░░██░░░░

Split Temporal (CORRECTO):
Train:  ████████████████░░░░░░░░
Test:   ░░░░░░░░░░░░░░░░████████
                         ↑
                    Punto de corte
```

## 🛠️ Herramientas para Split Temporal

### 1. Split Manual (Recomendado para simplicidad)

```python
# Basado en porcentaje
split_idx = int(len(df) * 0.7)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
```

```python
# Basado en fecha específica
cutoff_date = '2023-06-01'
train = df[df['fecha'] < cutoff_date]
test = df[df['fecha'] >= cutoff_date]
```

### 2. TimeSeriesSplit (Para validación cruzada)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Entrenar y evaluar
```

**Ventaja**: Permite validación cruzada respetando el orden temporal.

**Cómo funciona**:
```
Split 1: [Train: ████░░░░░░░░] [Test: ░░░░█░░░░░░░]
Split 2: [Train: █████░░░░░░░] [Test: ░░░░░█░░░░░░]
Split 3: [Train: ██████░░░░░░] [Test: ░░░░░░█░░░░░]
Split 4: [Train: ███████░░░░░] [Test: ░░░░░░░█░░░░]
Split 5: [Train: ████████░░░░] [Test: ░░░░░░░░█░░░]
```

Cada test siempre es posterior a su correspondiente train.

## 📊 Comparación de Resultados

### Resultados Típicos

| Métrica | Split Aleatorio | Split Temporal | Diferencia |
|---------|----------------|----------------|------------|
| R² | 0.92 | 0.68 | -26% |
| MAE | 8.5 | 14.2 | +67% |
| RMSE | 11.3 | 18.7 | +65% |

El split aleatorio **sobreestima** significativamente el desempeño.

## 🎓 Casos de Uso Donde Esto es Crítico

1. **Predicción de ventas/demanda**
   - Forecasting de inventario
   - Planificación de producción

2. **Trading algorítmico**
   - Predicción de precios de acciones
   - Backtesting de estrategias

3. **Detección de fraude**
   - Transacciones ordenadas temporalmente
   - Patrones que evolucionan con el tiempo

4. **Mantenimiento predictivo**
   - Logs de sensores en máquinas
   - Predicción de fallos

5. **Marketing digital**
   - Predicción de conversiones
   - Comportamiento de usuarios en el tiempo

6. **Forecasting climático**
   - Predicción de temperatura
   - Pronóstico de lluvias

## ⚠️ Señales de Alerta

Debes sospechar de data leakage temporal si:

1. **R² muy alto** (>0.95) en problemas complejos de series de tiempo
2. **Gran diferencia** entre métricas de validación y producción
3. **Features con información futura** (ej: promedios que incluyen el valor a predecir)
4. **Lag features calculados incorrectamente** (usando valores futuros)

## 🔍 Otras Consideraciones

### Gap entre Train y Test

En algunos casos, puedes querer dejar un **gap** (espacio) entre train y test:

```python
# Dejar 7 días de gap
train_end = int(len(df) * 0.7)
gap = 7
test_start = train_end + gap

X_train = X.iloc[:train_end]
X_test = X.iloc[test_start:]
```

**Razón**: Simular latencia en obtención de datos o evitar dependencias inmediatas.

### Walk-Forward Validation

Para sistemas de trading y producción continua:

```python
# Entrenar con ventana móvil
window_size = 365  # 1 año
for i in range(window_size, len(df)):
    train_data = df.iloc[i-window_size:i]
    test_point = df.iloc[i]
    # Entrenar, predecir, evaluar
```

## 🚀 Cómo Usar Este Proyecto

### Paso 1 — Instala las dependencias

Las de esta demo ya vienen con la base del repo. Desde la **raíz del repo**:

```bash
uv sync --extra dev
```

### Paso 2 — Abre el notebook

```bash
cd projects/d2-temporal-split
uv run jupyter lab data_leakage_temporal.ipynb
```

### Paso 3 — Qué hacer en clase

Ejecuta las celdas en orden y observa cómo las métricas del split
aleatorio son artificialmente optimistas comparadas con las del split
temporal.

**Objetivo pedagógico (20 min):** entender que el *shuffle* del split
aleatorio es correcto en datos iid, pero rompe la realidad en datos
con componente temporal.

## 📝 Contenido del Notebook

1. **Generación de datos sintéticos** de ventas con tendencia y estacionalidad
2. **Split aleatorio (incorrecto)** y evaluación
3. **Visualización del data leakage** en el tiempo
4. **Split temporal (correcto)** y evaluación
5. **Comparación de métricas** entre ambos enfoques
6. **Uso de TimeSeriesSplit** para validación cruzada temporal

## 💡 Conclusiones Clave

1. **NUNCA uses `train_test_split` con `shuffle=True` en datos temporales**
2. **Respeta siempre el orden cronológico**: train = pasado, test = futuro
3. **Las métricas optimistas son peligrosas** - prefiere métricas conservadoras y realistas
4. **Usa `TimeSeriesSplit`** para validación cruzada en series de tiempo
5. **Simula producción**: tu split debe reflejar cómo usarás el modelo en la vida real

## 📚 Referencias

- [Scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Avoiding Data Leakage in Time Series](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Time Series Cross-Validation](https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9)

## 👨‍🏫 Para Instructores

Este material es ideal para:
- Cursos de Machine Learning aplicado
- Módulos sobre validación de modelos
- Talleres sobre series de tiempo
- Discusiones sobre buenas prácticas en ML

**Duración estimada**: 60-75 minutos

**Conceptos relacionados**:
- Feature engineering temporal
- Lag features
- Walk-forward validation
- Backtesting

---

**Licencia**: MIT License
**Autor**: David Palacio Jiménez

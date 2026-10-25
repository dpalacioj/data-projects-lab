# Estructura de Datos

Este directorio sigue la convención de **cookiecutter-data-science** para organizar datos en diferentes etapas de procesamiento.

## Subdirectorios

### `raw/` - Datos Crudos (Originales)

**Propósito**: Almacenar datos originales, inmutables y sin procesar.

**Contenido actual**:
- `automoviles_usados.parquet` (10,000 registros, 15 columnas)
- `automoviles_usados.csv` (versión CSV de los mismos datos)

**Regla de oro**: Los datos en `raw/` **NUNCA** deben modificarse. Son la fuente de verdad.

**Git tracking**: Este proyecto **SÍ** incluye los datos raw/ en el repositorio para fines educativos. En proyectos reales, normalmente los datos raw/ se ignoran en .gitignore.

---

### `interim/` - Datos Intermedios

**Propósito**: Almacenar datos que han pasado por transformaciones intermedias pero que aún no están en su forma final para modelado.

**Ejemplos de lo que podría guardarse aquí**:
- Datos después de limpieza inicial (valores nulos eliminados)
- Datos después de eliminar outliers
- Datos después de fusionar múltiples fuentes
- Datos después de filtrar registros no válidos

**¿Por qué está vacío en este proyecto?**

En este proyecto, **no generamos archivos interim/** porque:

1. **Pipeline en memoria**: Usamos pipelines de sklearn que procesan los datos en memoria desde raw/ hasta el modelo final, sin pasos intermedios guardados.

2. **Datos limpios desde el inicio**: Los datos en `raw/` ya están limpios y no requieren transformaciones intermedias pesadas.

3. **Simplicidad educativa**: Para proyectos de enseñanza, es más simple mantener todo el procesamiento en código sin guardar pasos intermedios.

**Cuándo usar `interim/`**:
- Cuando el procesamiento es muy costoso y quieres guardar checkpoints
- Cuando trabajas con múltiples fuentes de datos que necesitas fusionar
- Cuando el proceso de limpieza es complejo y quieres validar pasos intermedios

**Ejemplo de uso**:
```python
# Si quisieras guardar datos intermedios:
import pandas as pd

# 1. Cargar datos raw
df = pd.read_parquet('data/raw/automoviles_usados.parquet')

# 2. Limpieza básica
df_clean = df.dropna()
df_clean = df_clean[df_clean['precio'] > 0]

# 3. Guardar en interim/
df_clean.to_parquet('data/interim/automoviles_limpiados.parquet')
```

---

### `processed/` - Datos Procesados (Listos para Modelado)

**Propósito**: Almacenar datos en su forma final, completamente procesados y listos para entrenar modelos.

**Ejemplos de lo que podría guardarse aquí**:
- Features engineering aplicado (nuevas columnas calculadas)
- Datos después de codificación one-hot de categóricas
- Datos después de normalización/estandarización
- Train/test splits guardados
- Matrices de features finales

**¿Por qué está vacío en este proyecto?**

En este proyecto, **no generamos archivos processed/** porque:

1. **Sklearn Pipelines**: Usamos `sklearn.pipeline.Pipeline` que aplica el preprocessing automáticamente cuando entrenamos o predecimos. No necesitamos guardar los datos procesados porque el pipeline los genera on-the-fly.

2. **Eficiencia de espacio**: Los datos procesados (especialmente después de one-hot encoding) pueden ser muy grandes. Generarlos on-demand ahorra espacio en disco.

3. **Consistencia**: Al procesar siempre desde raw/ con el mismo pipeline, garantizamos que train y test usen exactamente las mismas transformaciones.

**Cuándo usar `processed/`**:
- Cuando el preprocessing es muy costoso computacionalmente
- Cuando quieres compartir datos listos para modelado con otros
- Cuando trabajas con datasets enormes y quieres evitar reprocesar
- Cuando el feature engineering es complejo y quieres validar las features finales

**Ejemplo de uso**:
```python
# Si quisieras guardar datos procesados:
from preprocessing import load_data, create_preprocessor, prepare_features
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Cargar y preparar datos
df = load_data()
X, y = prepare_features(df, include_target=True)

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Aplicar preprocessing
preprocessor = create_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 4. Guardar como arrays
import joblib
joblib.dump((X_train_processed, X_test_processed, y_train, y_test),
            'data/processed/train_test_split.pkl')
```

---

## Flujo de Datos en Este Proyecto

```
data/raw/automoviles_usados.parquet
         ↓
    (load_data)
         ↓
    DataFrame
         ↓
    (prepare_features)
         ↓
    X, y
         ↓
    (train_test_split)
         ↓
    X_train, X_test
         ↓
    (Pipeline: preprocessing + modelo)
         ↓
    Predicciones
```

**Todo ocurre en memoria**, por eso `interim/` y `processed/` están vacíos.

---

## Comparación: Con vs Sin Archivos Intermedios

### Enfoque actual (Sin archivos intermedios)

**Ventajas**:
- Más simple para entender
- Menos archivos que gestionar
- Garantiza que siempre se usa el mismo preprocessing
- Ideal para proyectos educativos y datasets pequeños-medianos

**Desventajas**:
- Reprocesa los datos cada vez que entrenas
- Puede ser lento para datasets muy grandes

### Enfoque con archivos intermedios

**Ventajas**:
- Más rápido para datasets grandes (procesas una vez, usas muchas veces)
- Útil para debugging de pasos de procesamiento
- Facilita compartir datos procesados con el equipo

**Desventajas**:
- Más complejo de mantener
- Usa más espacio en disco
- Riesgo de desincronización entre raw y processed

---

## Cuándo Usar Cada Enfoque

### Usa enfoque SIN archivos intermedios (como este proyecto):
- Dataset < 1GB
- Preprocessing rápido (< 1 minuto)
- Proyecto educativo o prototipo
- Trabajas solo o en equipo pequeño

### Usa enfoque CON archivos intermedios:
- Dataset > 10GB
- Preprocessing lento (> 5 minutos)
- Proyecto en producción
- Equipo grande que necesita compartir datos procesados
- Debugging complejo de feature engineering

---

## Referencias

- [Cookiecutter Data Science - Data Organization](https://cookiecutter-data-science.drivendata.org/)
- [Best Practices for Data Science Project Structure](https://drivendata.github.io/cookiecutter-data-science/)

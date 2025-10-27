# Quickstart Guide - Predicción de Precios de Automóviles

Este documento proporciona los comandos exactos para poner en marcha el proyecto rápidamente.

## Pre-requisitos

- Python 3.11 o superior
- uv instalado
- Repositorio clonado

## Pasos de Ejecución Rápida

### IMPORTANTE: Instalación de Dependencias

Las dependencias deben instalarse **UNA VEZ** desde la **raíz del repositorio** (donde está el `pyproject.toml` principal):

```bash
# Ir a la raíz del repositorio (NO al subdirectorio del proyecto)
cd /path/to/data-projects-lab

# Instalar dependencias (solo una vez)
uv sync --extra regresion-automoviles
```

**Verificar instalación:**
```bash
uv run python -c "import mlflow, gradio, fastapi; print('Dependencias OK')"
```

Si ves "Dependencias OK", puedes continuar. Si no, ejecuta de nuevo `uv sync --extra regresion-automoviles` desde la raíz.

### 1. Navegar al Proyecto y Verificar Setup

Ahora sí, ve al proyecto (todos los comandos siguientes se ejecutan desde aquí):

```bash
cd projects/prediccion-precios-automoviles
```

**OPCIONAL: Verificar que todo funcione**
```bash
uv run python test_setup.py
```

Este script verifica:
- Todas las dependencias están instaladas
- Los datos existen (si no, te dice cómo generarlos)
- El modelo existe (si no, te dice cómo entrenarlo)
- Los módulos funcionan correctamente
- Se puede hacer una predicción de prueba

Si todo está OK (✓), puedes saltar directamente al paso 3 (Gradio/FastAPI). Si no, sigue con el paso 2.

**Nota:** Los datos ya están incluidos en `data/raw/automoviles_usados.parquet` (10,000 registros con 15 columnas). Si por alguna razón no están, contacta al instructor.

### 2. Entrenar Modelo

```bash
uv run python train_quick.py
```

**Salida esperada:**
- `models/random_forest_model.pkl` (25 MB)
- `models/random_forest_metadata.json`
- R² ≈ 0.95 en conjunto de prueba

### 3. Probar Predicción

```bash
uv run python -c "
import sys
sys.path.append('src')
from predict import load_model, predict_single, create_example_input, format_prediction

model = load_model()
input_data = create_example_input()
precio = predict_single(model, input_data)
print(f'Precio predicho: {format_prediction(precio)}')
"
```

### 4. Ejecutar Aplicación Gradio (Interfaz Web)

```bash
uv run python app_gradio.py
```

**Abrir en navegador:** http://localhost:7860

### 5. Ejecutar API FastAPI

```bash
uv run uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

**Documentación interactiva:** http://localhost:8000/docs

**Probar API con curl:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "marca": "Toyota",
           "tipo_carroceria": "SUV",
           "año": 2020,
           "kilometraje": 50000,
           "tipo_combustible": "Gasolina",
           "transmision": "Automática",
           "cilindrada": 2000,
           "potencia": 150,
           "peso": 1500,
           "consumo": 8.5,
           "color": "Blanco",
           "edad_propietarios": 1,
           "calificacion_estado": 8.5,
           "region_venta": "Centro"
         }'
```

## Verificación del Proyecto

### Verificar que todo funciona:

```bash
# Verificar datos
ls -lh data/raw/

# Verificar modelo
ls -lh models/

# Verificar estructura
tree -L 2 -I '__pycache__|*.pyc'
```

### Estadísticas del Proyecto

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/raw/automoviles_usados.parquet')
print(f'Registros: {len(df):,}')
print(f'Columnas: {len(df.columns)}')
print(f'Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
print(f'Precio promedio: \${df.precio.mean():,.2f}')
print(f'Precio min: \${df.precio.min():,.2f}')
print(f'Precio max: \${df.precio.max():,.2f}')
"
```

## Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'mlflow'" (o gradio, fastapi, etc.)

**Causa:** Dependencias no instaladas o instaladas desde el lugar incorrecto

**Solución:**
```bash
# 1. Ir a la RAÍZ del repositorio (NO al subdirectorio del proyecto)
cd /path/to/data-projects-lab

# 2. Verificar que estás en el lugar correcto (debe existir pyproject.toml)
ls pyproject.toml

# 3. Instalar dependencias
uv sync --extra regresion-automoviles

# 4. Verificar instalación
uv run python -c "import mlflow, gradio, fastapi; print('OK')"

# 5. Ahora sí, ir al proyecto
cd projects/prediccion-precios-automoviles

# 6. Ejecutar scripts
uv run python train_quick.py
```

**Nota importante:**
- `uv sync` debe ejecutarse desde la **raíz del repositorio** donde está `pyproject.toml`
- Los scripts (`train_quick.py`, etc.) se ejecutan desde el **directorio del proyecto**
- Usa siempre `uv run python` para ejecutar scripts, no solo `python`

### Error: "No such file or directory"
**Causa:** No estás en el directorio correcto
**Solución:**
```bash
# Verificar en qué directorio estás
pwd

# Deberías ver algo como: .../data-projects-lab/projects/prediccion-precios-automoviles
# Si no, navega al lugar correcto
cd /path/to/data-projects-lab/projects/prediccion-precios-automoviles
```

### Error: "Modelo no encontrado"
**Causa:** No has entrenado el modelo
**Solución:**
```bash
uv run python train_quick.py
```

### Error: "Datos no encontrados"
**Causa:** No has generado los datos
**Solución:**
```bash
uv run python generate_data.py
```

### Puerto ocupado (Gradio/FastAPI)
**Solución:** Cambia el puerto:
```bash
# Gradio
uv run python app_gradio.py --server-port 7861

# FastAPI
uv run uvicorn app_fastapi:app --port 8001
```

## Comandos de Limpieza

```bash
# Limpiar datos generados
rm -rf data/raw/*

# Limpiar modelos
rm -rf models/*.pkl models/*.json

# Limpiar notebooks ejecutados
rm -rf notebooks/*_ejecutado.ipynb

# Limpiar caché
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Scripts Disponibles

- `generate_data.py`: Genera dataset sintético de 10,000 automóviles
- `train_quick.py`: Entrena modelo Random Forest sin MLFlow
- `app_gradio.py`: Interfaz web interactiva
- `app_fastapi.py`: API REST

## Estructura de Archivos Generados

```
projects/prediccion-precios-automoviles/
├── data/
│   └── raw/
│       ├── automoviles_usados.csv        # 10,000 registros, 1.1 MB
│       └── automoviles_usados.parquet    # 10,000 registros, 285 KB
└── models/
    ├── random_forest_model.pkl          # Modelo entrenado, 25 MB
    └── random_forest_metadata.json      # Metadatos del modelo
```

## Métricas Esperadas

El modelo entrenado debería alcanzar aproximadamente:

- **R² (Test):** 0.95
- **RMSE (Test):** $1,300
- **MAE (Test):** $630
- **Tiempo de entrenamiento:** 10-30 segundos
- **Tamaño del modelo:** 25 MB

## Próximos Pasos

1. Explorar los notebooks Jupyter para análisis detallado
2. Experimentar con la interfaz Gradio
3. Probar la API con diferentes requests
4. Modificar hiperparámetros en `src/config.py`
5. Entrenar con MLFlow: `uv run python -c "import sys; sys.path.append('src'); from training import train_model; train_model('random_forest', use_mlflow=True)"`

## Recursos

- **README principal:** `README.md`
- **Documentación cookiecutter:** https://cookiecutter-data-science.drivendata.org
- **FastAPI docs:** http://localhost:8000/docs (cuando la API está corriendo)
- **Notebooks:** `notebooks/`

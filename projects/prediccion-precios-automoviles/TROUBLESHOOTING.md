# Guía de Solución de Problemas

Esta guía te ayudará a resolver los problemas más comunes al usar este proyecto.

## Problema 1: "ModuleNotFoundError: No module named 'mlflow'"

### Síntomas
```
Traceback (most recent call last):
  File ".../train_quick.py", line 13, in <module>
    from training import train_model
  File ".../src/training.py", line 18, in <module>
    import mlflow
ModuleNotFoundError: No module named 'mlflow'
```

### Causa
Las dependencias **NO** están instaladas, o las instalaste desde el lugar incorrecto.

### Solución Paso a Paso

#### Paso 1: Entender la estructura del repositorio

Este proyecto es parte de un **monorepo** (un repositorio con múltiples proyectos):

```
data-projects-lab/                 <- RAÍZ del repositorio
├── pyproject.toml                 <- Configuración de dependencias AQUÍ
├── .venv/                         <- Entorno virtual AQUÍ
├── projects/
│   ├── prediccion-precios-automoviles/  <- PROYECTO (donde estás ahora)
│   │   ├── data/raw/              <- Datos del proyecto (ya incluidos)
│   │   ├── train_quick.py
│   │   └── src/
│   └── otros-proyectos/
└── tutorials/
```

#### Paso 2: Ir a la RAÍZ del repositorio

```bash
# Si estás en: .../prediccion-precios-automoviles/
cd ../..

# Verificar que estás en la raíz
pwd
# Deberías ver: /path/to/data-projects-lab

ls pyproject.toml
# Deberías ver: pyproject.toml
```

#### Paso 3: Instalar dependencias

```bash
uv sync --extra regresion-automoviles
```

Esto instala:
- pandas, numpy, scikit-learn
- plotly
- mlflow
- gradio
- fastapi, uvicorn
- jupyter

#### Paso 4: Verificar instalación

```bash
uv run python -c "import mlflow, gradio, fastapi, plotly; print('Todas las dependencias instaladas correctamente')"
```

Si ves el mensaje de éxito, continúa al siguiente paso.

#### Paso 5: Volver al proyecto

```bash
cd projects/prediccion-precios-automoviles
```

#### Paso 6: Ejecutar script

```bash
uv run python train_quick.py
```

### ¿Por qué pasa esto?

`uv` busca el archivo `pyproject.toml` para saber qué dependencias instalar. Si ejecutas `uv sync` desde el subdirectorio del proyecto, no encuentra el `pyproject.toml` correcto.

**Regla simple:**
- `uv sync` → Ejecutar en la **RAÍZ** (donde está `pyproject.toml`)
- `uv run python script.py` → Ejecutar en el **PROYECTO** (donde está el script)

---

## Problema 2: "No such file or directory"

### Síntomas
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/automoviles_usados.parquet'
```

### Causa
Los datos no están en la ubicación esperada o estás en el directorio incorrecto.

### Solución

```bash
# Verificar directorio actual
pwd
# Deberías estar en: .../prediccion-precios-automoviles

# Si no estás ahí:
cd /path/to/data-projects-lab/projects/prediccion-precios-automoviles

# Verificar que existen los datos
ls -lh data/raw/

# Si no hay datos, contacta al instructor
# Los datos deben estar incluidos con el proyecto
```

---

## Problema 3: "Modelo no encontrado"

### Síntomas
```
FileNotFoundError: No se encontraron modelos entrenados
```

### Causa
No has entrenado ningún modelo.

### Solución

```bash
# Entrenar modelo
uv run python train_quick.py

# Verificar que se creó
ls -lh models/
```

---

## Problema 4: Puerto ya en uso (Gradio/FastAPI)

### Síntomas
```
OSError: [Errno 48] Address already in use
```

### Causa
Ya hay otra aplicación usando el puerto 7860 (Gradio) o 8000 (FastAPI).

### Solución

**Opción 1: Matar el proceso**
```bash
# Para Gradio (puerto 7860)
lsof -ti:7860 | xargs kill -9

# Para FastAPI (puerto 8000)
lsof -ti:8000 | xargs kill -9
```

**Opción 2: Usar otro puerto**
```bash
# Gradio en puerto 7861
uv run python app_gradio.py --server-port 7861

# FastAPI en puerto 8001
uv run uvicorn app_fastapi:app --port 8001
```

---

## Problema 5: "Cannot import name 'load_model' from 'predict'"

### Síntomas
```
ImportError: cannot import name 'load_model' from 'predict'
```

### Causa
Python no encuentra el módulo `predict.py` o hay un conflicto de nombres.

### Solución

```bash
# Verificar que estás en el proyecto
pwd
# Deberías ver: .../prediccion-precios-automoviles

# Verificar que existe el módulo
ls src/predict.py

# Probar import
uv run python -c "import sys; sys.path.append('src'); from predict import load_model; print('OK')"
```

---

## Problema 6: Jupyter no encuentra el kernel

### Síntomas
Al abrir un notebook, no aparece el kernel o da error.

### Solución

```bash
# Desde la RAÍZ del repositorio
cd /path/to/data-projects-lab

# Instalar kernel
uv run python -m ipykernel install --user --name=data-projects-lab

# Abrir Jupyter
cd projects/prediccion-precios-automoviles
uv run jupyter notebook notebooks/
```

---

## Problema 7: "uv: command not found"

### Síntomas
```
bash: uv: command not found
```

### Causa
`uv` no está instalado.

### Solución

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# O con homebrew
brew install uv

# Verificar instalación
uv --version
```

---

## Problema 8: Versión de Python incompatible

### Síntomas
```
error: No interpreter found for Python >=3.11
```

### Causa
No tienes Python 3.11 o superior instalado.

### Solución

```bash
# Verificar versión actual
python3 --version

# Instalar Python 3.11+ con uv
uv python install 3.11

# O con pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

---

## Comandos Útiles de Diagnóstico

### Verificar todo está OK

```bash
# Script de diagnóstico completo
uv run python -c "
import sys
print(f'Python: {sys.version}')

try:
    import pandas, numpy, sklearn
    print('ML libs: OK')
except:
    print('ML libs: ERROR')

try:
    import plotly
    print('Plotly: OK')
except:
    print('Plotly: ERROR')

try:
    import mlflow
    print('MLFlow: OK')
except:
    print('MLFlow: ERROR')

try:
    import gradio
    print('Gradio: OK')
except:
    print('Gradio: ERROR')

try:
    import fastapi
    print('FastAPI: OK')
except:
    print('FastAPI: ERROR')

import os
print(f'Directorio: {os.getcwd()}')

import pathlib
if pathlib.Path('data/raw/automoviles_usados.parquet').exists():
    print('Datos: OK')
else:
    print('Datos: NO GENERADOS')

if pathlib.Path('models/random_forest_model.pkl').exists():
    print('Modelo: OK')
else:
    print('Modelo: NO ENTRENADO')
"
```

### Limpiar y empezar de nuevo

```bash
# Ir a la raíz
cd /path/to/data-projects-lab

# Limpiar entorno virtual
rm -rf .venv

# Reinstalar todo
uv sync --extra regresion-automoviles

# Ir al proyecto
cd projects/prediccion-precios-automoviles

# Limpiar solo modelos (NO eliminar datos, están incluidos con el proyecto)
rm -rf models/*.pkl models/*.json

# Regenerar modelo
uv run python train_quick.py

# NOTA: Si accidentalmente borraste data/raw/, contacta al instructor
# Los datos deben estar incluidos con el proyecto
```

---

## ¿Todavía tienes problemas?

### Información útil para reportar bugs

Cuando reportes un problema, incluye:

1. **Comando que ejecutaste:**
   ```
   uv run python train_quick.py
   ```

2. **Directorio donde lo ejecutaste:**
   ```
   pwd
   /Users/tu-usuario/data-projects-lab/projects/prediccion-precios-automoviles
   ```

3. **Error completo:**
   ```
   Copia el error completo del terminal
   ```

4. **Versiones:**
   ```bash
   uv --version
   python3 --version
   ```

5. **Salida del diagnóstico:**
   ```bash
   uv run python -c "import sys; print(sys.version); import pandas; print('OK')"
   ```

---

## Preguntas Frecuentes

### ¿Por qué usar `uv run` en lugar de `python`?

`uv run` garantiza que se use el entorno virtual correcto con todas las dependencias instaladas. `python` solo usa el Python del sistema.

### ¿Puedo usar `pip` en lugar de `uv`?

Sí, pero no es recomendado. Este proyecto está configurado para `uv`. Si insistes:

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar desde requirements generado
uv pip compile pyproject.toml -o requirements.txt
pip install -r requirements.txt
```

### ¿Dónde se guardan los archivos generados?

```
prediccion-precios-automoviles/
├── data/raw/              <- Datos generados
├── models/                <- Modelos entrenados
└── mlruns/               <- Experimentos de MLFlow (si usas MLFlow)
```

### ¿Cómo actualizo las dependencias?

```bash
# Desde la raíz
cd /path/to/data-projects-lab
uv sync --extra regresion-automoviles --upgrade
```

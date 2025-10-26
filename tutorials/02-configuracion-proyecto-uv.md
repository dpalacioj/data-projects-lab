# Tutorial: Configuración de Proyecto con uv

## 📚 Introducción

`uv` es una herramienta moderna y ultra-rápida para gestión de paquetes Python, escrita en Rust. Reemplaza a pip, pip-tools, pipx, poetry, pyenv, virtualenv y más, ofreciendo una experiencia unificada y eficiente.

### Ventajas de usar uv:
- ⚡ **10-100x más rápido** que pip y pip-tools
- 🔒 **Archivos .lock reproducibles** para garantizar consistencia
- 🐍 **Gestión automática de versiones de Python**
- 📦 **Resolución de dependencias inteligente**
- 🛠️ **Todo en una herramienta** (no necesitas pip, venv, etc.)

## 🚀 Paso 1: Instalación de uv

### macOS/Linux
```bash
# Método recomendado: usando el instalador oficial
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternativa con Homebrew (macOS)
brew install uv
```

### Windows
```powershell
# PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# O descarga el instalador desde:
# https://github.com/astral-sh/uv/releases
```

### Verificar instalación
```bash
uv --version
# Output esperado: uv 0.x.x
```

## 📁 Paso 2: Inicializar el Proyecto

### Crear estructura del proyecto
```bash
# Navegar al directorio del proyecto
cd data-projects-lab

# Inicializar proyecto con uv
uv init --name data-projects-lab --no-readme

# Esto creará:
# - pyproject.toml (configuración del proyecto)
# - .python-version (versión de Python a usar)
```

### Estructura resultante
```
data-projects-lab/
├── pyproject.toml       # Configuración del proyecto
├── .python-version      # Versión de Python (ej: 3.12)
├── tutorials/          
├── projects/           
├── examples/           
└── datasets/           
```

## 📝 Paso 3: Configurar pyproject.toml

### Configuración básica
```toml
[project]
name = "data-projects-lab"
version = "0.1.0"
description = "Notebooks educativos para IA y Analytics"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "David Palacio Jiménez", email = "tu-email@ejemplo.com"}
]

[project.urls]
Homepage = "https://github.com/tu-usuario/data-projects-lab"
Repository = "https://github.com/tu-usuario/data-projects-lab.git"
```

## 📦 Paso 4: Agregar Dependencias

### Dependencias principales para ciencia de datos
```bash
# Núcleo de ciencia de datos
uv add pandas numpy scipy

# Visualización
uv add matplotlib seaborn plotly

# Machine Learning
uv add scikit-learn

# Deep Learning (opcional)
uv add torch torchvision  # PyTorch
# O
uv add tensorflow  # TensorFlow

# Notebooks
uv add jupyter notebook ipykernel

# Utilidades
uv add python-dotenv tqdm
```

### Dependencias de desarrollo (solo para desarrollo)
```bash
# Herramientas de desarrollo
uv add --dev pytest pytest-cov
uv add --dev black isort flake8
uv add --dev mypy
uv add --dev ipdb  # Debugger para notebooks
```

### El pyproject.toml resultante
```toml
[project]
name = "data-projects-lab"
version = "0.1.0"
description = "Notebooks educativos para IA y Analytics"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "David Palacio Jiménez", email = "tu-email@ejemplo.com"}
]
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.18.0",
    "scikit-learn>=1.3.0",
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "ipykernel>=6.25.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.66.0"
]

[project.urls]
Homepage = "https://github.com/tu-usuario/data-projects-lab"
Repository = "https://github.com/tu-usuario/data-projects-lab.git"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "ipdb>=0.13.13"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## 🔒 Paso 5: El archivo uv.lock

### ¿Qué es uv.lock?
- Archivo que **congela** todas las versiones exactas de dependencias
- Garantiza reproducibilidad entre diferentes máquinas
- Se genera automáticamente al usar `uv add` o `uv sync`
- **DEBE** ser incluido en control de versiones (git)

### Generar/Actualizar el lock file
```bash
# Sincronizar y generar uv.lock
uv sync

# Esto:
# 1. Lee pyproject.toml
# 2. Resuelve todas las dependencias
# 3. Genera/actualiza uv.lock
# 4. Instala los paquetes en .venv
```

### Estructura del uv.lock (ejemplo parcial)
```toml
version = 1
requires-python = ">=3.9"

[[package]]
name = "numpy"
version = "1.26.4"
source = { registry = "https://pypi.org/simple" }
...

[[package]]
name = "pandas"
version = "2.2.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "numpy" },
    { name = "python-dateutil" },
    { name = "pytz" },
    { name = "tzdata" },
]
...
```

## 🔧 Paso 6: Comandos Esenciales de uv

### Gestión de dependencias
```bash
# Agregar una dependencia
uv add requests

# Agregar dependencia de desarrollo
uv add --dev pre-commit

# Actualizar una dependencia específica
uv add pandas@latest

# Eliminar una dependencia
uv remove requests

# Ver dependencias instaladas
uv pip list
```

### Sincronización y entornos
```bash
# Sincronizar entorno con pyproject.toml y uv.lock
uv sync

# Sincronizar incluyendo dependencias de desarrollo
uv sync --dev

# Crear entorno limpio desde cero
rm -rf .venv
uv sync
```

### Ejecutar comandos en el entorno
```bash
# Ejecutar Python
uv run python script.py

# Ejecutar Jupyter
uv run jupyter notebook

# Ejecutar pytest
uv run pytest

# O activar el entorno manualmente
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

## 🎯 Paso 7: Configurar .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Entorno virtual de uv
.venv/
venv/
ENV/

# uv
.uv/
uv.lock.bak

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Datos y modelos (si son muy grandes)
*.csv
*.xlsx
*.h5
*.pkl
*.joblib
models/
data/large/

# Configuración local
.env
.env.local

# Pytest
.pytest_cache/
.coverage
htmlcov/
*.cover

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json
```

## 📋 Paso 8: Flujo de Trabajo Completo

### Para iniciar el proyecto por primera vez:
```bash
# 1. Clonar repositorio
git clone <url-del-repo>
cd data-projects-lab

# 2. Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sincronizar entorno desde uv.lock
uv sync --dev

# 4. Verificar instalación
uv run python --version
uv run jupyter --version

# 5. Iniciar Jupyter
uv run jupyter notebook
```

### Para colaboradores del proyecto:
```bash
# 1. Clonar repo
git clone <url-del-repo>
cd data-projects-lab

# 2. Sincronizar (uv.lock garantiza mismas versiones)
uv sync --dev

# 3. ¡Listo para trabajar!
uv run jupyter notebook
```

### Para agregar nuevas dependencias:
```bash
# 1. Agregar dependencia
uv add streamlit

# 2. Commit cambios
git add pyproject.toml uv.lock
git commit -m "Add streamlit dependency"

# 3. Push
git push
```

## 🚨 Resolución de Problemas Comunes

### Problema: "command not found: uv"
```bash
# Verificar instalación
which uv

# Re-instalar si es necesario
curl -LsSf https://astral.sh/uv/install.sh | sh

# Agregar al PATH (si no se agregó automáticamente)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc  # o ~/.bashrc
source ~/.zshrc
```

### Problema: Conflictos de dependencias
```bash
# Limpiar y re-sincronizar
rm -rf .venv
rm uv.lock
uv sync
```

### Problema: Jupyter no encuentra el kernel
```bash
# Instalar kernel en Jupyter
uv run python -m ipykernel install --user --name=data-projects-lab

# Seleccionar el kernel en Jupyter Notebook
```

## 📚 Recursos Adicionales

- [Documentación oficial de uv](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 621 - pyproject.toml](https://peps.python.org/pep-0621/)

## ✅ Checklist Final

- [ ] uv instalado y funcionando
- [ ] pyproject.toml configurado con dependencias
- [ ] uv.lock generado y commiteado
- [ ] .gitignore actualizado
- [ ] Entorno virtual creado (.venv/)
- [ ] Jupyter notebook funcionando
- [ ] README actualizado con instrucciones

---

**Siguiente paso:** Crear tu primer notebook en `tutorials/` y empezar a desarrollar contenido educativo.
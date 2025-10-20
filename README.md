# 🚀 Laboratorio de Proyectos de Datos
## Especialización en Inteligencia Artificial y Analítica

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Bienvenido al repositorio oficial del Laboratorio de Proyectos de Datos. Este espacio está diseñado para estudiantes que desean aprender desarrollo moderno en Data Science y Machine Learning con herramientas de IA integradas.

## 📖 Contenido del Repositorio

Este repositorio contiene una compilación educativa de notebooks, tutoriales y proyectos completos que cubren el ciclo completo de Machine Learning, desde la exploración de datos hasta el deployment de modelos.

### 📚 Tutoriales Disponibles

Explora la carpeta **[`tutorials/`](./tutorials/)** para acceder a todos los tutoriales paso a paso. El contenido está en constante crecimiento y abarca:

- **Preprocesamiento de Datos**: Técnicas de imputación, encoding, scaling, PCA y feature engineering
- **Validación y Optimización**: Cross-validation, GridSearch, RandomSearch, Optuna
- **Interpretabilidad de Modelos**: Feature Importance, SHAP, LIME
- **AutoML**: Frameworks como PyCaret y FLAML
- **Y mucho más...**

> 💡 Los tutoriales se actualizan regularmente. Revisa la carpeta `tutorials/` para ver todo el contenido disponible.

### 🚀 Proyectos Completos

La carpeta **[`projects/`](./projects/)** contiene implementaciones end-to-end de proyectos reales:

#### 🚢 Proyecto Titanic
Predicción de supervivencia con el dataset del Titanic. Incluye:
- Descarga y limpieza de datos
- Análisis exploratorio (EDA)
- Entrenamiento con AutoML (PyCaret y FLAML)
- Interpretabilidad de modelos con SHAP
- Deployment con Streamlit

*Más proyectos en desarrollo...*

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas |
|-----------|-------------|
| **Data Science** | Pandas, NumPy, Scikit-learn |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **AutoML** | PyCaret, FLAML |
| **Interpretabilidad** | SHAP, LIME *(opcional)* |
| **Optimización** | Optuna, GridSearch *(opcional)* |
| **Deployment** | Streamlit |
| **Experimentación** | MLflow, W&B *(opcional)* |
| **Notebooks** | Jupyter Lab |
| **Asistente IA** | Claude Code |
| **Gestión de Paquetes** | uv (recomendado), pip |
| **Control de Versiones** | Git/GitHub |

## 🚀 Cómo Empezar

### Prerrequisitos
- Git instalado
- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recomendado para gestión de paquetes)
- Claude Desktop (opcional pero recomendado para desarrollo asistido con IA)

### Instalación

#### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/data-projects-lab.git
cd data-projects-lab
```

#### 2️⃣ Configurar el entorno con uv (Recomendado)
```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar el entorno (instala todas las dependencias)
uv sync
```

#### 3️⃣ Instalar dependencias opcionales (según tus necesidades)
```bash
# Para interpretabilidad de modelos (SHAP, LIME)
uv pip install -e ".[interpretability]"

# Para experimentación avanzada (Optuna, MLflow, W&B)
uv pip install -e ".[experiment]"

# Para desarrollo y testing
uv pip install -e ".[dev]"

# O instalar todo de una vez
uv pip install -e ".[all]"
```

#### 4️⃣ Verificar la instalación
```bash
# Listar paquetes instalados
uv pip list

# Ejecutar Jupyter Lab
uv run jupyter lab
```

### Método Alternativo (sin uv)
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias base
pip install pandas scikit-learn matplotlib seaborn jupyter pycaret flaml streamlit
```

## 📚 Estructura del Repositorio

```
data-projects-lab/
├── 📝 README.md                   # Este archivo
├── 📄 CLAUDE.md                   # Configuración para Claude Code
├── 📄 LICENSE                     # Licencia MIT
├── 🔧 .claude/                   # Configuraciones locales
│   └── settings.local.json
├── 📁 tutorials/                  # Tutoriales paso a paso
├── 📁 projects/                   # Proyectos completos
├── 📁 examples/                   # Ejemplos de código
└── 📁 datasets/                   # Datos para ejercicios
```

## 💻 Uso

### Ejecutar Jupyter Notebooks
```bash
# Con uv (recomendado)
uv run jupyter lab

# O activando el entorno virtual
source .venv/bin/activate
jupyter lab
```

### Ejecutar Proyectos con Streamlit
Ejemplo con el proyecto Titanic:
```bash
# Desde la raíz del repositorio
uv run streamlit run projects/titanic/08_titanic_streamlit.py
```

## 🎓 Metodología de Aprendizaje

### Filosofía del Curso
- ✅ **Aprender haciendo**: Cada concepto se practica inmediatamente con código ejecutable
- ✅ **IA como copiloto**: Claude Code acelera el aprendizaje y ayuda a resolver problemas
- ✅ **Proyectos reales**: Aplicaciones prácticas de la industria con datasets públicos
- ✅ **Código reproducible**: Todos los notebooks están probados y documentados

### 📍 Rutas de Aprendizaje

#### Para Principiantes
1. Explora los tutoriales básicos en `tutorials/`
2. Comienza con el proyecto Titanic en `projects/titanic/`
3. Practica modificando los notebooks existentes

#### Para Estudiantes Avanzados
1. Revisa los tutoriales de optimización e interpretabilidad
2. Implementa tus propios modelos en los proyectos
3. Experimenta con AutoML y técnicas avanzadas
4. Crea deployments con Streamlit

## 🎯 Características Principales

- ✅ **Código Limpio y Documentado**: Notebooks estructurados con explicaciones detalladas
- ✅ **Entorno Reproducible**: Gestión de dependencias con `pyproject.toml` y `uv`
- ✅ **Educativo**: Explicaciones paso a paso, desde conceptos básicos hasta avanzados
- ✅ **Práctico**: Proyectos completos con datasets reales
- ✅ **Moderno**: Uso de AutoML (PyCaret, FLAML), interpretabilidad (SHAP, LIME)
- ✅ **Deployment Ready**: Ejemplos de puesta en producción con Streamlit
- ✅ **Asistido por IA**: Integración con Claude Code para aprendizaje acelerado

## 📝 Desarrollo de Contenido

Al crear nuevos notebooks o tutoriales, sigue estas pautas:

- Estructura notebooks con objetivos de aprendizaje claros
- Incluye explicaciones en markdown entre celdas de código
- Proporciona datasets de ejemplo o código para generarlos
- Agrega ejercicios donde sea apropiado
- Usa nombres de variables claros y comentarios educativos
- Prueba todas las celdas para asegurar reproducibilidad

## 🤝 Contribuciones

Este es un repositorio educativo. Si encuentras errores o tienes sugerencias:

1. Abre un [Issue](https://github.com/tu-usuario/data-projects-lab/issues) describiendo el problema/mejora
2. Si quieres contribuir código, crea un Pull Request
3. Asegúrate de seguir las guías de estilo del proyecto

## 👤 Autor

**David Palacio Jiménez**

- 📧 Email: david.palacio@example.com
- 🐙 GitHub: [@tu-usuario](https://github.com/tu-usuario)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

**Copyright (c) 2025 David Palacio Jiménez**

## 🙏 Agradecimientos

- [Anthropic](https://www.anthropic.com/) por Claude Code
- Comunidad de Python y Data Science
- Contribuidores de scikit-learn, pandas, PyCaret, FLAML y todas las librerías open source
- Estudiantes que proporcionan feedback valioso

---

## 💡 Consejos para Estudiantes

> **Usa Claude Code mientras sigues los tutoriales**: Pregúntale sobre el código, pídele que explique conceptos, que te ayude a debuggear o que sugiera mejoras. ¡Es tu tutor personal de IA disponible 24/7!

### Preguntas Frecuentes

**¿Necesito experiencia previa en Python?**
Se recomienda conocimiento básico de Python, pero los tutoriales incluyen explicaciones desde cero.

**¿Puedo usar estos notebooks para mis propios proyectos?**
¡Sí! Todo el código está bajo licencia MIT. Úsalo, modifícalo y compártelo.

**¿Cómo reporto un error o sugiero mejoras?**
Abre un [Issue](https://github.com/tu-usuario/data-projects-lab/issues) en GitHub con todos los detalles.

**¿Se agregarán más tutoriales?**
Sí, el repositorio se actualiza regularmente con nuevo contenido.

---

⭐️ **Si este repositorio te resulta útil, considera darle una estrella en GitHub!**

**¡Feliz aprendizaje! 🚀📊🤖**
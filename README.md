# Proyecto I — Laboratorio de Proyectos de Datos

**Universidad de Medellín** — Especialización en Ciencia de Datos e Inteligencia Artificial

> **Revisión del material:** 2026-10-20 · Python 3.11 · gestor de paquetes `uv`.

Este repositorio es el material de clase del módulo **Proyecto I**. El hilo
conductor son cuatro sesiones de cuatro horas que recorren el ciclo CRISP-DM
sobre un caso real (recompra sobre Online Retail II), y una colección de
proyectos y tutoriales de apoyo que el estudiante puede consultar fuera de
clase.

Este NO es un curso de MLOps. Las piezas pesadas de MLOps (orquestación,
despliegue en la nube, monitoreo) se ven en el módulo siguiente. Lo que aquí se
practica es la **disciplina previa**: formular el objetivo de análisis, mirar
los datos con honestidad, construir pipelines sin fuga, entrenar modelos
reproducibles y dejar un entregable que otra persona pueda recoger.

---

## Empieza aquí

### Paso 0 — Diagnostica tu máquina

**Antes de instalar nada.** Este script corre con cualquier Python 3.11
o superior, sin una sola dependencia, y te dice exactamente qué falta:

```bash
git clone https://github.com/dpalacioj/data-projects-lab.git
cd data-projects-lab
python3 scripts/smoke_test.py      # en Windows: python scripts\smoke_test.py
```

Va a salir en rojo si no tienes `uv` o Python 3.11, y está bien: todavía no
instalaste nada. Lo que importa es **qué** dice en rojo. Cada `FAIL` trae el
comando que lo arregla.

### Paso 1 — Instala `uv`

`uv` gestiona Python y las dependencias. Sin esto no arranca nada.

| SO | Comando |
|---|---|
| macOS / Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

Cierra y reabre la terminal después de instalar para que `uv` aparezca en el `PATH`.

### Paso 2 — Prepara el ambiente del curso

```bash
make setup       # instala base + extra dev
make smoke       # verifica que todo quedó bien
```

Sin `make` (Windows sin GnuWin32), los equivalentes son:

```bash
uv sync --extra dev
uv run python scripts/smoke_test.py
```

**No sigas hasta que `make smoke` salga limpio.** La columna de la derecha de
cada verificación trae el comando para arreglarla.

---

## Cómo navegar el repositorio

El material de clase vive en [`sesiones/`](sesiones/). **Es la primera carpeta
que debes abrir.** El resto son soportes.

Orden de lectura recomendado para el estudiante:

1. **Este README** → dónde estás, cómo correrlo.
2. **[`sesiones/README.md`](sesiones/README.md)** → mapa de las cuatro sesiones y cómo se leen.
3. **La sesión del día** → `sesiones/sXX-tema/README.md`. Es la guía para seguir la clase.
4. **El caso guía** → [`projects/segmentacion-clientes/`](projects/segmentacion-clientes/). Es donde el código vive. Las sesiones S02 y S03 enlazan a sus notebooks y a su paquete `src/segmentacion/`.
5. **[`tutorials/`](tutorials/)** → refuerzo transversal (preprocesamiento, validación cruzada, interpretabilidad, etc.). Se leen cuando aparece el tema; no son prerrequisito.
6. **Otros proyectos en [`projects/`](projects/)** → ejemplos completos (Titanic, Mercado Libre, regresión de automóviles). Son material histórico de inspiración. No se tocan en clase.

Orden para el instructor:

1. [`sesiones/_templates/CONVENCIONES.md`](sesiones/_templates/CONVENCIONES.md) → reglas de estilo del material.
2. [`sesiones/sXX-tema/README.md`](sesiones/) → el guión completo (encuadre, bloques, tiempos).
3. [`sesiones/sXX-tema/checklist-dia-de-clase.md`](sesiones/) → qué revisar 15 minutos antes de entrar.
4. [`docs/specs/`](docs/specs/) → contratos de cifras que las sesiones citan.

---

## Las cuatro sesiones

| # | Sesión | Pregunta que responde | Unidad | Estado |
|---|---|---|---|---|
| 1 | [Objetivo de análisis y CRISP-DM](sesiones/s01-objetivo-crisp-dm/) | ¿Cómo decides qué proyecto hacer y por qué, antes de abrir el IDE? | U1 | Placeholder |
| 2 | [Exploración de datos con Polars](sesiones/s02-exploracion-polars/) | ¿Qué de este archivo representa un hecho del mundo, y qué es un artefacto de cómo se cargó? | U2 | Completa |
| 3 | [Procesamiento, pipelines y modelos](sesiones/s03-procesamiento-sklearn/) | ¿Cómo entrenas un modelo sin que la métrica mienta? | U3 | Completa |
| 4 | [Aplicación y entrega](sesiones/s04-app-mlops/) | ¿Cómo cierras el proyecto para que otra persona pueda recogerlo? | U4 | Placeholder |

Cada sesión arranca mostrando el problema antes que la herramienta, trae su
`README.md` como guión, uno o más `.md` temáticos (teoría + práctica), un
`taller.md` para la aplicación individual y un `troubleshooting.md` con los
errores reales que ya vimos en clase.

---

## El caso guía

**Online Retail II** (UCI 502, licencia CC BY 4.0). Un año de facturas de un
retailer online. La pregunta del caso: **¿quién va a volver a comprar en los
próximos N días?** Da naturalmente:

- Cohortes temporales reales (no sintéticas).
- Fuga temporal verificable (si construyes la variable después de la fecha de corte, el modelo "acierta" por tramposo).
- Variables RFM (recencia, frecuencia, monto) como punto de partida.
- Un lugar legítimo para usar un LLM como transformador de texto (categorizar descripciones de productos).

El código del caso vive en
[`projects/segmentacion-clientes/`](projects/segmentacion-clientes/). Los
contratos de cifras (qué número tiene que salir en cada paso) están en
[`docs/specs/`](docs/specs/). Si una cifra del material no coincide, se
re-captura la **spec**, no se maquilla la prueba.

---

## Qué hay en cada carpeta

| Carpeta | Qué contiene |
|---|---|
| [`sesiones/`](sesiones/) | El material de clase. Cada sesión trae su `README.md` (guión), `.md` temáticos (teoría + práctica), `taller.md` (aplicación individual) y `troubleshooting.md` (errores reales ya vistos) |
| [`projects/segmentacion-clientes/`](projects/segmentacion-clientes/) | El caso guía. Es el proyecto en el que trabajan S02, S03 y S04: notebooks del flujo CRISP-DM, paquete `src/segmentacion/` instalable, tests de contrato de cifras, app Streamlit |
| [`projects/`](projects/) | Otros proyectos completos, independientes del caso guía. Material histórico de referencia: Titanic (AutoML), Mercado Libre (clasificación con XGBoost), regresión de automóviles (MLflow + FastAPI). No se tocan en clase |
| [`tutorials/`](tutorials/) | Guías transversales que no caben completas en una sesión: preprocesamiento, métricas de clasificación vs regresión, validación cruzada, interpretabilidad, MLflow básico, MCP, recursos de IA |
| [`docs/specs/`](docs/specs/) | Contratos de cifras del caso guía. Son el ancla de los tests `-m slow` y de las tablas que aparecen en los `.md` de sesión. Si falla una cifra, se investiga aquí |
| [`scripts/`](scripts/) | Herramientas sueltas del repositorio. Hoy solo `smoke_test.py` (el diagnóstico del entorno) |
| [`datasets/`](datasets/) | Datasets de los proyectos legacy (Mercado Libre). Los datos del caso guía no se versionan; se descargan con `scripts/descargar_datos.py` dentro del proyecto |
| [`imagenes/`](imagenes/) | Figuras usadas en tutoriales (data leakage, data drift, autoML). Se referencian por path relativo |

Y en la raíz: `pyproject.toml` y `uv.lock` declaran y fijan las dependencias;
`Makefile` es la lista de comandos (ejecutar `make` sin argumentos los lista);
`.github/workflows/ci.yml` corre lint y tests en cada push.

---

## Ambiente y dependencias

Un solo `pyproject.toml` en la raíz. La organización:

- **Base** (`[project.dependencies]`): pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter. Lo que **cualquier** sesión necesita.
- **Extras por proyecto** (`[project.optional-dependencies]`): dependencias específicas de cada caso. Se instalan solo cuando trabajas en ese proyecto.
- **Extra `dev`**: pytest, ruff, mypy, ipython.

Flujo típico:

```bash
uv sync --extra dev                                     # base del curso + herramientas
uv sync --extra segmentacion-clientes --extra dev       # + lo del caso guía (S02/S03)
uv sync --extra titanic                                 # proyecto Titanic (histórico)
uv sync --extra regresion-automoviles                   # proyecto autos (histórico, MLflow+FastAPI)
```

**Por qué así.** Los proyectos legacy traen deps pesadas (MLflow, Gradio,
FastAPI, XGBoost, LightGBM) y el estudiante que solo necesita Polars no debería
descargar 2 GB para empezar. Mantener extras por proyecto declara
explícitamente qué pide cada uno, y deja el ambiente base ligero.

**Para añadir un proyecto nuevo** (p. ej. redes recurrentes):

1. Crear `projects/<nombre>/` con su propio `README.md`.
2. Añadir una entrada en `[project.optional-dependencies]` en `pyproject.toml`:

   ```toml
   rnn = [
       "torch>=2.0",
       "lightning>=2.0",
   ]
   ```

3. Documentar en el `README.md` del proyecto cómo instalarlo (`uv sync --extra rnn`).

---

## Instructor

Las notas que no debe ver el estudiante van en `instructor/` (hoy vacío; se
usará para guiones privados cuando aparezcan). El guión público de cada clase
es el `README.md` de la sesión correspondiente.

---

## Licencia

MIT. Ver [`LICENSE`](LICENSE).

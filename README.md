---
title: Inicio
nav_order: 1
permalink: /
---
# Proyecto I — Laboratorio de Proyectos de Datos

Material de clase del módulo **Proyecto I** de la Especialización en
Ciencia de Datos e Inteligencia Artificial, Universidad de Medellín.

Este módulo no es de MLOps. Las piezas pesadas (orquestación, despliegue
en la nube, monitoreo) se ven en el módulo siguiente. Aquí practicamos
la disciplina previa: formular bien el objetivo, mirar los datos con
honestidad, construir pipelines sin fuga, entrenar modelos reproducibles
y dejar un entregable que otra persona pueda recoger sin pedir ayuda.

Si llegas por primera vez, lee este README hasta el final antes de abrir
nada. No asume que ya sabes `uv`, ni git, ni `Makefile`. Es más corto
que una clase.

Última revisión: **2026-10-25** · Python 3.11 · gestor `uv`.

---

## Cómo está organizado el repo

Tres carpetas principales. Cada una tiene su propia lógica:

- **`sesiones/`** es la columna vertebral: el guión de clase. Lo que se
  dicta en las 4 horas de cada sesión.
- **`tutorials/`** son temas sueltos (git, uv, métricas, interpretabilidad,
  ...). Se leen cuando aparecen en una sesión, o cuando el estudiante
  quiere repasar algo puntual.
- **`projects/`** es donde el código vive de verdad. El caso guía del
  curso, cuatro proyectos didácticos de distinta dificultad, y cuatro
  demos cortas para conceptos puntuales.

Las sesiones citan tutoriales y aplican sobre proyectos. El flujo normal
es: abres la sesión del día, ella te dice qué tutorial repasar y en qué
proyecto practicar.

---

## Antes de empezar

Vas a necesitar tres cosas:

1. **Python 3.11 o superior.**
2. **El gestor de paquetes `uv`.** Instalación:
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. **Una cuenta de GitHub.**

Cierra y reabre la terminal después de instalar `uv` (para que entre al
`PATH`). Verifica con `uv --version`.

### Diagnóstico sin instalar nada

El script `scripts/smoke_test.py` corre con Python puro, sin dependencias,
y te dice qué falta en tu máquina. Córrelo antes de instalar:

```bash
git clone https://github.com/dpalacioj/data-projects-lab.git
cd data-projects-lab
python3 scripts/smoke_test.py
```

**Si `python3` no te funciona**, no estás haciendo nada mal: el nombre del
intérprete depende de cómo instalaste Python. Prueba en este orden hasta
que uno responda:

- `python3 --version` (lo común en macOS y Linux)
- `python --version` (algunos Linux, y Windows con la casilla "Add Python to PATH" marcada)
- `py --version` (Windows con el instalador oficial)

Usa el que te funcione en todos los comandos del README. Si ninguno
responde, Python no está instalado y debes instalarlo primero
(descarga en [python.org](https://www.python.org/downloads/)).

Si `smoke_test.py` sale en rojo, cada `FAIL` trae el comando que lo
arregla. No te asustes.

### Instalar la base del curso

```bash
make setup       # base del curso + herramientas dev
make smoke       # verifica que todo quedó bien
```

(Sin `make`, los equivalentes son `uv sync --extra dev` y
`uv run python scripts/smoke_test.py`.)

No sigas hasta que `make smoke` salga limpio.

### Aprende git antes de tocar código

Si nunca has abierto un Pull Request, lee
[`tutorials/01-git-flujo-de-trabajo.md`](tutorials/01-git-flujo-de-trabajo.md).
Son 15 minutos y resuelve el 90% de los problemas que vemos en clase a
la hora de entregar.

---

## Qué leer en tu primera semana

En este orden. No es un calendario, es la secuencia que tiene sentido.

1. Este README hasta el final (estás aquí).
2. [`tutorials/00-ambientes-venv-conda-uv/`](tutorials/00-ambientes-venv-conda-uv/) — por qué `uv` y no otro gestor.
3. [`tutorials/01-git-flujo-de-trabajo.md`](tutorials/01-git-flujo-de-trabajo.md) — clonar, rama, commit, PR, CI (visión general).
4. [`tutorials/01b-tu-primer-ci.md`](tutorials/01b-tu-primer-ci.md) — hands-on: construye tu propio repo con CI desde cero. Teoría de qué es, por qué, y práctica paso a paso con diagramas. Es el que más dudas quita.
5. [`sesiones/s01-objetivo-crisp-dm/README.md`](sesiones/s01-objetivo-crisp-dm/README.md) — el encuadre del curso: CRISP-DM y dónde se practica cada fase.
6. [`sesiones/README.md`](sesiones/README.md) — mapa de las sesiones de clase.
7. [`trabajo-final/README.md`](trabajo-final/README.md) — qué vas a entregar al final del módulo (léelo temprano para saber hacia dónde vas).

De ahí en adelante, cada sesión te lleva a los tutoriales y proyectos
que necesite.

---

## Las sesiones y las clases

El repo organiza el material en **4 sesiones conceptuales** (una por
unidad del microcurrículo). El módulo se dicta en **8 clases presenciales
de 4 horas** (32 horas de interacción + 16 de trabajo autónomo por unidad).
Dos clases por sesión conceptual:

| Sesión | Unidad | Clases presenciales | Material |
|---|---|---|---|
| S01 · Encuadre + CRISP-DM | U1 | Clase 1 y 2 | [`sesiones/s01-objetivo-crisp-dm/`](sesiones/s01-objetivo-crisp-dm/) |
| S02 · Exploración con Polars | U2 | Clase 3 y 4 | [`sesiones/s02-exploracion-polars/`](sesiones/s02-exploracion-polars/) |
| S03 · Procesamiento, pipelines, modelos | U3 | Clase 5 y 6 | [`sesiones/s03-procesamiento-sklearn/`](sesiones/s03-procesamiento-sklearn/) |
| S04 · Aplicación y entregable | U4 | Clase 7 y 8 | [`sesiones/s04-app-mlops/`](sesiones/s04-app-mlops/) + [`projects/04-reconocimiento-imagenes/`](projects/04-reconocimiento-imagenes/) |

### Calendario detallado (8 clases × 4 horas)

El desglose asume que una clase de 4 horas no es monólogo: hay
calentamiento, pausa, taller en vivo y Q&A. Los tiempos son
aproximados; la realidad varía según la cohorte.

| Clase | Unidad | De qué va | Material del repo | Desglose aproximado de las 4h |
|---|---|---|---|---|
| **1** | U1 | Encuadre del curso + ambiente + git | S01 + tutoriales 00 y 01 | 30' bienvenida y presentaciones · 1h30' qué es CRISP-DM y por qué · 15' break · 1h taller: `make setup` + primer commit en una rama · 45' Q&A y asignación de tarea |
| **2** | U1 | Formulación individual: aterrizar los 6 campos de CRISP-DM fase 1 | S01 + notebook `00_objetivo_de_analisis.ipynb` del caso guía | 20' repaso + dudas · 1h30' trabajo guiado llenando la ficha de fase 1 sobre un caso propio · 15' break · 1h30' presentación corta de cada ficha + feedback entre pares · 25' cierre |
| **3** | U2 | Exploración: carga, perfilado, limpieza con Polars | S02 (bloques 1-3) + notebooks 01-03 del caso guía | 20' calentamiento + preguntas pendientes · 2h teoría + práctica (perfilado y 5 reglas de limpieza) · 15' break · 1h taller individual sobre un dataset propio · 25' Q&A |
| **4** | U2 | EDA (manual vs. automatizado) + construcción de la cohorte | S02 (bloques 4-5) + notebooks 04-05 + demo [`d1-ejercicio-preprocesamiento`](projects/d1-ejercicio-preprocesamiento/) | 20' revisión del taller · 1h30' EDA + preparación de variables RFM · 15' break · 1h30' taller: construir la cohorte sobre un dataset propio · 25' cierre |
| **5** | U3 | Partición temporal, pipelines sin fuga, feature engineering | S03 bloques A-B + notebooks 06-07 + demo [`d2-temporal-split`](projects/d2-temporal-split/) | 20' arranque · 2h teoría + práctica (particiones A/B, `ColumnTransformer`, fuga temporal) · 15' break · 1h taller · 25' Q&A |
| **6** | U3 | Modelos, métricas, umbral, guardado del artefacto; LLM como transformador (opcional) | S03 bloques C-E + notebook 09 + notebook 08 (LLM, opcional) + demo [`d4-umbrales-ejemplo`](projects/d4-umbrales-ejemplo/) | 20' repaso · 2h entrenamiento, comparación de modelos, tabla de umbrales, guardado con `metadatos.json` · 15' break · 1h taller individual (quien quiera, prueba el bloque E con FakeLLM) · 25' cierre |
| **7** | U4 | Del modelo al entregable: Streamlit, FastAPI, Gradio | S04 + [`projects/04-reconocimiento-imagenes/`](projects/04-reconocimiento-imagenes/) + [`projects/03-prediccion-precios-automoviles/`](projects/03-prediccion-precios-automoviles/) | 20' arranque · 1h30' demo en vivo del flujo notebook → paquete → app · 15' break · 1h30' taller: envolver un modelo propio en Streamlit · 25' Q&A |
| **8** | U4 | Taller final: presentaciones del entregable individual + cierre | Entregable propio + proyecto 04 como referencia | 20' arranque · 2h30' presentaciones (≈15' por estudiante) · 15' break · 45' feedback cruzado y cierre del módulo |

### Si estudias por tu cuenta (sin asistir a las clases)

Si usas el repo como material autodidacta, un ritmo razonable es ~5 horas
por semana durante 7 semanas:

| Semana | Qué cubres | Material |
|---|---|---|
| **1** | Fundamentos: ambiente, git, uv, Makefile, encuadre CRISP-DM | Tutoriales 00-03 + S01 completa + notebook `00_objetivo_de_analisis` del caso guía (CRISP-DM fase 1 aplicada) |
| **2** | Exploración de datos (parte 1) | S02 bloques 1-3 + notebooks 01-03 del caso guía |
| **3** | Exploración de datos (parte 2) | S02 bloques 4-5 + notebooks 04-05 + demo d1 |
| **4** | Procesamiento y pipelines sin fuga | S03 bloques A-B + notebooks 06-07 + demo d2 |
| **5** | Modelos, métricas, umbrales, artefacto, LLM (opcional) | S03 bloques C-E + notebook 09 + notebook 08 (LLM, opcional) + demo d4 |
| **6** | Aplicación: notebook → paquete → app | S04 + proyecto 04 (CNN) corrido completo |
| **7** | Entregable propio | Aplicas lo aprendido sobre un dataset tuyo, usando el proyecto 05 como plantilla |

> **Nota.** El notebook `10_segmentacion_no_supervisada` del caso guía
> (clustering K-Means + DBSCAN) es una vía paralela al flujo supervisado
> de las clases. Si te interesa el enfoque no supervisado, ábrelo tras la
> semana 5 como extensión opcional.

Si tienes más tiempo o quieres contrastar enfoques, abre después los
proyectos didácticos 01 (titanic, AutoML) y 03 (automóviles, MLflow +
FastAPI). Son material histórico pero útiles.

---

## El caso guía

**Online Retail II** (UCI, licencia CC BY 4.0): un año de facturas de un
retailer online. La pregunta es: **¿quién va a volver a comprar en los
próximos 90 días?**

El caso vive en [`projects/00-segmentacion-clientes/`](projects/00-segmentacion-clientes/)
y las sesiones S02 y S03 trabajan sobre él. Si solo abres un proyecto,
abre ese.

---

## Los demás proyectos

Los proyectos están numerados por nivel. Un `ls projects/` los muestra
en el orden que yo los enseño:

```
00-segmentacion-clientes       Caso guía
01-titanic                     AutoML, primer contacto
02-clasificador-meli           XGBoost
03-prediccion-precios-autos    Modularización + MLflow + FastAPI
04-reconocimiento-imagenes     CNN + Streamlit + FastAPI (cierra el flujo)
05-ejemplo-completo            Plantilla del trabajo final (clasificador tabular + CI + Makefile propio)
d1-ejercicio-preprocesamiento  Demo: pipelines sin fuga
d2-temporal-split              Demo: split temporal
d3-promociones-usuarios        Demo: construcción de target
d4-umbrales-ejemplo            Demo: umbrales y ROC
```

El detalle con "cuándo abrir cada uno" está en
[`projects/README.md`](projects/README.md).

El proyecto **04-reconocimiento-imagenes** es el que cierra el ciclo que
S04 pide: notebook paso a paso → código modular en `src/` → app Streamlit
→ API FastAPI. Si quieres ver *cómo se entrega un modelo*, es ese.

El proyecto **05-ejemplo-completo** es la plantilla específica del
**trabajo final del módulo**. Trae todo lo del 04 más `tests/`,
`Makefile` propio y una plantilla de CI (`ci-ejemplo.yml`) lista para
copiar a tu propio repositorio. Si tu proyecto final es tabular, cloná
su estructura y adaptala a tu problema.

---

## Los tutoriales

El [`tutorials/README.md`](tutorials/README.md) tiene el índice ordenado.
En resumen:

- **00-03** — fundamentos: ambiente, git, proyecto `uv`, `Makefile`.
- **04-06** — notebooks como experimentación → paquete entregable.
- **07-12** — temas transversales: preprocesamiento, métricas, validación, interpretabilidad, MLflow + FastAPI.
- **90-92** — complementos opcionales (Claude Code, MCP, catálogo de recursos). Si te interesa, léelos; no son requisito.

---

## Integración continua

Cuando abres un PR, GitHub ejecuta un robot
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) que corre
`make lint` y `make test` en un servidor neutral. Si pasa, check verde ✅.
Si falla, check rojo ❌ con el log del error.

Si falla en CI pero pasa en tu laptop, casi siempre es un archivo que
olvidaste commitear. El tutorial de git lo explica con detalle en su
[Paso 6](tutorials/01-git-flujo-de-trabajo.md#paso-6--qué-pasa-cuando-abres-el-pr).

**Para entender CI de verdad**, no basta con usar el de este repo. Haz
el [tutorial 01b](tutorials/01b-tu-primer-ci.md): construyes tu propio
repo con CI desde cero, con teoría, diagramas y 12 pasos prácticos. Al
terminar vas a saber qué significa cada línea del `ci.yml` y qué pasa
exactamente cuando abres un PR.

No tienes que configurar el CI de este repo. Ya está.

---

## Qué hay en cada carpeta (referencia rápida)

| Carpeta | Qué contiene |
|---|---|
| `sesiones/` | Guiones de clase. S02 y S03 completas; S01 y S04 son documentos de marco |
| `tutorials/` | Temas sueltos, numerados por orden lógico de lectura |
| `projects/` | Casos de código, numerados por nivel |
| `scripts/` | Herramientas sueltas. Hoy solo `smoke_test.py` |
| `datasets/` | Datasets versionados de proyectos (p. ej. Meli). Los del caso guía no se versionan |
| `imagenes/` | Figuras usadas en tutoriales |
| `trabajo-final/` | Especificación del entregable final del módulo |

En la raíz: `pyproject.toml` y `uv.lock` declaran y fijan dependencias;
`Makefile` es la lista de comandos (`make` sin argumentos los muestra).

---

## Ambiente y dependencias

Un solo `pyproject.toml` en la raíz. Las dependencias base (pandas, numpy,
sklearn, matplotlib, jupyter) son lo que **cualquier** sesión necesita.
Cada proyecto declara sus dependencias específicas como **extra**:

```bash
uv sync --extra dev                                   # base del curso
uv sync --extra segmentacion-clientes --extra dev     # + caso guía (S02/S03)
uv sync --extra reconocimiento-imagenes --extra dev   # + proyecto 04 (CNN)
uv sync --extra titanic                               # + proyecto 01
uv sync --extra regresion-automoviles                 # + proyecto 03
```

Los nombres de los extras no llevan prefijo numérico (son nombres
lógicos, no de carpeta). La lista completa está en
[`projects/README.md`](projects/README.md).

**Por qué así.** El proyecto 03 trae MLflow y Gradio (deps pesadas); el 04
trae PyTorch (más pesado todavía). Un estudiante que solo necesita Polars
no debería descargar 2 GB para arrancar. Los extras por proyecto
mantienen la base ligera.

---

## Licencia

MIT. Ver [`LICENSE`](LICENSE).

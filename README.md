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

Si sale en rojo, cada `FAIL` trae el comando que lo arregla. No te asustes.

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
3. [`tutorials/01-git-flujo-de-trabajo.md`](tutorials/01-git-flujo-de-trabajo.md) — clonar, rama, commit, PR, CI.
4. [`sesiones/s01-objetivo-crisp-dm/README.md`](sesiones/s01-objetivo-crisp-dm/README.md) — el encuadre del curso: CRISP-DM y dónde se practica cada fase.
5. [`sesiones/README.md`](sesiones/README.md) — mapa de las sesiones de clase.

De ahí en adelante, cada sesión te lleva a los tutoriales y proyectos
que necesite.

---

## Las sesiones

| # | Tema | Estado |
|---|---|---|
| S01 | Encuadre: CRISP-DM como columna vertebral | 📄 marco (lectura, no presencial) |
| S02 | Exploración de datos con Polars | 🟢 completa |
| S03 | Procesamiento, pipelines y modelos | 🟢 completa |
| S04 | Aplicación: del modelo al entregable | 📄 marco (trabajo autónomo sobre el proyecto 04) |

S01 y S04 son documentos de marco, no sesiones de 4 horas. S01 la lees
antes de arrancar; S04 la practicas al final sobre tu propia app. S02 y
S03 son las sesiones presenciales con guión completo.

Detalle en [`sesiones/README.md`](sesiones/README.md).

---

## El caso guía

**Online Retail II** (UCI, licencia CC BY 4.0): un año de facturas de un
retailer online. La pregunta es: **¿quién va a volver a comprar en los
próximos 90 días?**

El caso vive en [`projects/00-segmentacion-clientes/`](projects/00-segmentacion-clientes/)
y las sesiones S02 y S03 trabajan sobre él. Si solo abres un proyecto,
abre ese. Los contratos de cifras (qué número tiene que salir en cada
paso) están en [`docs/specs/`](docs/specs/).

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

No tienes que configurar CI. Ya está.

---

## Qué hay en cada carpeta (referencia rápida)

| Carpeta | Qué contiene |
|---|---|
| `sesiones/` | Guiones de clase. S02 y S03 completas; S01 y S04 son documentos de marco |
| `tutorials/` | Temas sueltos, numerados por orden lógico de lectura |
| `projects/` | Casos de código, numerados por nivel |
| `docs/specs/` | Contratos de cifras del caso guía (ancla de los tests `-m slow`) |
| `scripts/` | Herramientas sueltas. Hoy solo `smoke_test.py` |
| `datasets/` | Datasets de proyectos legacy. Los del caso guía no se versionan |
| `imagenes/` | Figuras usadas en tutoriales |
| `instructor/` | Notas privadas del docente (no las ve el estudiante) |

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

---

Hecho con dolor de cabeza y café por **David Palacio Jiménez**
(`david.palacio@unosquare.com`) para la Universidad de Medellín.

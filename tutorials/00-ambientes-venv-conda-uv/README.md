# Tutorial 00 — Un mismo notebook con tres ambientes: `.venv`, miniconda y uv

> Revisión: **2026-10-20** · Python 3.11 · macOS M-series (los números cambian en Windows / Linux).

El objetivo de este tutorial **no es** aprender KNN. Es demostrar, con el mismo
notebook y las mismas librerías, cómo tres gestores de ambientes distintos
responden a la misma pregunta: _"déjame un ambiente listo para abrir este
notebook"_.

Al final el estudiante decide con criterio, no por moda, cuál usar.

---

## El notebook

[`knn_vecinos.ipynb`](knn_vecinos.ipynb). Cinco celdas de código:

1. Imports de numpy, pandas, scikit-learn y matplotlib (si alguno falla, el ambiente no quedó bien).
2. Carga del dataset iris (150 flores, 4 variables, 3 especies).
3. Partición 70/30 estratificada por clase.
4. Entrena un `KNeighborsClassifier(k=5)` e imprime accuracy + classification report.
5. Barre `k` de 1 a 15 y grafica la curva de accuracy.

Dependencias idénticas en los tres flujos: `numpy`, `pandas`, `scikit-learn`,
`matplotlib`, `jupyterlab`. Son ~400–700 MB una vez instalados (varía por
gestor). Nada exótico; el objetivo es que las tres rutas lleguen al mismo
resultado.

---

## Antes de empezar

Los tres flujos necesitan **Python 3.11**. Cómo tenerlo:

- **Con uv** (recomendado si todavía no tienes Python 3.11): `uv python install 3.11`.
- **Con el instalador oficial**: https://www.python.org/downloads/ y pedir Python 3.11.
- **Con miniconda**: `conda create -n py311 python=3.11` instala una copia aislada.

Verifica con `python3.11 --version` (debe salir `Python 3.11.x`).

---

## Cómo medir el tiempo

Antes de cada comando de instalación, antepón `time`. En macOS/Linux:

```bash
time (
  <los comandos que montan el ambiente>
)
```

En PowerShell (Windows):

```powershell
Measure-Command { <los comandos> }
```

Lo que importa es el **wall clock** (línea `real` en bash, `TotalSeconds` en
PowerShell): es lo que el estudiante efectivamente espera frente a la pantalla.

---

## Ruta A — `.venv` + `pip` (lo que viene con Python)

**Pro:** no requiere instalar nada extra; `venv` y `pip` vienen con Python.
**Contra:** sin lockfile; sin caché compartida entre proyectos; depende de que
los índices de PyPI estén rápidos ese día.

```bash
cd tutorials/00-ambientes-venv-conda-uv/

# Monta el ambiente y mídelo
time (
  python3.11 -m venv .venv
  .venv/bin/pip install -r requirements.txt
)

# Lanza jupyter desde el ambiente
.venv/bin/jupyter lab knn_vecinos.ipynb
```

En Windows:

```powershell
cd tutorials\00-ambientes-venv-conda-uv

Measure-Command {
  python3.11 -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
}

.venv\Scripts\jupyter lab knn_vecinos.ipynb
```

**Gotcha típico:** olvidar `source .venv/bin/activate` (o usar
`.venv/bin/pip` directamente) y terminar instalando en el Python del sistema.

---

## Ruta B — miniconda (`conda env create`)

**Pro:** maneja dependencias no-Python (binarios, CUDA, GDAL). Útil si el
proyecto necesita librerías compiladas con compiladores raros. **Contra:**
pesado; el resolvedor puede ser lento; el canal `defaults` está deprecado y hay
que usar `conda-forge`.

```bash
cd tutorials/00-ambientes-venv-conda-uv/

# Monta el ambiente y mídelo
time (
  conda env create -f environment.yml
)

conda activate knn-tutorial
jupyter lab knn_vecinos.ipynb
```

**Gotcha típico:** `conda env create` falla por conflictos de canales. El
archivo `environment.yml` ya fija `conda-forge`; si tu `~/.condarc` tiene
`defaults`, puede haber resolución más lenta o errores.

**Para repetir la medición** (si el ambiente ya existe):

```bash
conda env remove -n knn-tutorial -y
# y vuelve a correr `conda env create -f environment.yml`
```

---

## Ruta C — uv (`uv sync`)

**Pro:** rápido (resolver escrito en Rust, descargas en paralelo); lockfile
automático en `uv.lock`; caché global compartida entre proyectos. **Contra:**
es una herramienta nueva que hay que instalar la primera vez.

Instalar `uv` una sola vez (si no lo tienes):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh         # macOS/Linux
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Y luego:

```bash
cd tutorials/00-ambientes-venv-conda-uv/

# Monta el ambiente y mídelo
time (
  uv sync
)

uv run jupyter lab knn_vecinos.ipynb
```

**Gotcha típico:** confundir `uv pip install X` (modo "pip emulado", sin
lockfile) con `uv add X` (modo moderno, actualiza `pyproject.toml` + lockfile).
Para este tutorial usamos `uv sync`, que lee el `pyproject.toml` como contrato.

---

## Comparación

Medido en una MacBook Pro M-series con conexión doméstica, Python 3.11.14.
Los comandos son los que aparecen arriba, sin modificaciones.

| Gestor | Tiempo (ref. autor) | Tamaño `.venv` / env | Pasos del estudiante | Lockfile |
|---|---|---|---|---|
| `.venv` + `pip` (con caché pip) | **~51 s** | 570 MB | 2 comandos | No |
| miniconda (`conda env create`) | **~59 s** | 705 MB | 1 comando | No (hay `conda-lock` aparte) |
| uv con caché (segunda vez en adelante) | **~4 s** | 384 MB | 1 comando | Sí (`uv.lock` auto) |
| uv sin caché (primera vez de verdad) | **~7 s** | 384 MB | 1 comando | Sí |

**Mide la tuya:**

| Gestor | Tu tiempo | Tu máquina (SO / RAM / conexión) |
|---|---|---|
| `.venv` + `pip` | _____ s | _____ |
| miniconda | _____ s | _____ |
| uv | _____ s | _____ |

---

## Qué aprender de la comparación

1. **La velocidad no es el único punto.** uv es rápido, pero el regalo real es
   el `uv.lock` automático: dos estudiantes con el mismo repositorio obtienen
   las **mismas versiones exactas** de cada librería, sin pensarlo. Con `pip`
   hay que fijar manualmente (`pip freeze > requirements.txt`), y con `conda`
   hay que usar `conda-lock` aparte.
2. **conda tiene un nicho legítimo.** Si tu proyecto necesita binarios no-Python
   (CUDA toolkit, GDAL, Rdkit, etc.), `conda-forge` suele tener los wheels
   listos. Para proyectos puramente Python, es overkill.
3. **`.venv` es la línea base: siempre funciona.** Si no tienes `uv` ni
   `miniconda` y no puedes instalarlos (por política corporativa, por ejemplo),
   el `python3.11 -m venv` que trae Python te saca del aprieto.
4. **El ambiente del curso usa uv** porque el repositorio tiene 7 proyectos y
   el caché global de uv evita bajar los mismos ~400 MB una y otra vez.

---

## Qué sigue

Confirmaste que tu ambiente queda bien montado. Ya puedes seguir con el
[`tutorial 02 — configuración del proyecto con uv`](../02-configuracion-proyecto-uv.md),
que entra al patrón real del repositorio (base + extras por proyecto), o abrir
directamente el material de la [sesión S02](../../sesiones/s02-exploracion-polars/).

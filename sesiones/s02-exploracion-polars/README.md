---
title: "S02 · Exploración con Polars"
parent: Sesiones
nav_order: 2
has_children: true
---
# Sesión S02 · Exploración de datos con Polars

Última revisión: 2026-10-19. Versiones verificadas: Python 3.11.14,
Polars 2.0.0, scikit-learn 1.9.1, Pydantic 2.11, fg-data-profiling 4.20,
skimpy 0.0.21. La sesión se apoya en
[`projects/00-segmentacion-clientes/`](../../projects/00-segmentacion-clientes/).

La pregunta que organiza la sesión es:

> _«¿Qué de este archivo representa un hecho del mundo y qué de este archivo
> es un artefacto de cómo se cargó?»_ — porque la respuesta decide qué
> limpieza es honesta y qué limpieza es inventar realidad.

---

## 0. Para antes de clase

- [ ] Revisar [`08-checklist-dia-de-clase.md`](08-checklist-dia-de-clase.md).
- [ ] `uv sync --extra segmentacion-clientes --extra dev`.
- [ ] `uv run python projects/00-segmentacion-clientes/scripts/descargar_datos.py`
      (verifica el hash SHA-256 del zip de UCI 502).
- [ ] `uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_referencia.py`
      verde (confirma que el día de hoy las cifras de referencia se
      reproducen contra el Parquet real).

## 1. Objetivos de aprendizaje

Al terminar la sesión el estudiante puede:

1. **Formular** el objetivo de análisis a partir de una pregunta de
   negocio (CRISP-DM fase 1), con criterio de éxito analítico y de
   negocio escritos.
2. **Decidir** cuándo usar Polars y cuándo pandas, y leer un dataset
   grande declarando esquema (`schema_overrides`).
3. **Diferenciar** un nulo faltante de un nulo estructural y traducir
   esa diferencia a reglas de limpieza.
4. **Construir** las 5 reglas de limpieza como funciones separadas y
   encadenarlas con `.pipe`, registrando `n_antes/n_después` por regla.
5. **Reproducir** cuartiles, cercas de Tukey, correlaciones de Pearson
   y Spearman, y una comparación entre EDA manual y automatizado (skim
   + fg-data-profiling).
6. **Dejar** `HALLAZGOS.md` como insumo contractual para la Unidad 3.

## 2. Organización temporal (4 h de interacción · 8 h de trabajo independiente)

| Bloque | Duración | Contenido | Documento |
|---|---|---|---|
| **El problema** | 10 min | Tres cifras contradictorias del mismo dataset según cómo se cargó | este README §3 |
| **A — CRISP-DM fase 1** | 20 min | Objetivo de análisis + criterios de éxito | [`01-crisp-dm-fase1.md`](01-crisp-dm-fase1.md) |
| **B — Polars y por qué** | 30 min | Lectura con esquema, cuándo pandas | [`02-polars-vs-pandas.md`](02-polars-vs-pandas.md) · [`03-esquema-al-leer.md`](03-esquema-al-leer.md) |
| **C — Las 5 reglas** | 60 min | `preparar_ventas` + HALLAZGOS como contrato | [`04-limpieza-5-reglas.md`](04-limpieza-5-reglas.md) |
| **D — EDA manual vs automatizado** | 60 min | Tukey, correlaciones, skim, fg-profiling | [`05-eda-manual-vs-auto.md`](05-eda-manual-vs-auto.md) |
| **Taller** | 60 min | Aplicar a un dataset propio del estudiante | [`06-taller.md`](06-taller.md) |

## 3. El problema (antes de las herramientas)

Antes de abrir cualquier IDE, tres preguntas en voz alta:

1. _«Si yo cuento las facturas que empiezan por `C` sin leer el Excel
   con el esquema declarado, obtengo **cero**. Si lo leo con esquema,
   obtengo **19 494**. ¿Cuál cifra es la verdadera?»_
   > El motor adivinó `Invoice` como entero y descartó silenciosamente
   > las que no eran parseables. La "cifra verdadera" **no existe**
   > hasta que no se decide cómo leer.

2. _«El dataset tiene 243 007 filas sin `Customer ID`. ¿Imputo la mediana?»_
   > Trampa. Esas filas tienen todas `Price = 0`, sin `Description` o
   > con `Description` nula. **No son ventas**: son ajustes de inventario.
   > Imputar es inventar realidad. La respuesta correcta es `exigir_cliente`
   > y declarar la pérdida.

3. _«Pearson entre `Quantity` y `Price` es −0,005; Spearman es −0,428.
   ¿Qué variable copio al pipeline de U3?»_
   > Ni una ni la otra: la pregunta está mal planteada. La respuesta
   > vive en U3 y pide `log1p` + `StandardScaler`.

Estas tres preguntas motivan los tres bloques siguientes.

## 4. Materiales y artefactos

- [`01-crisp-dm-fase1.md`](01-crisp-dm-fase1.md) — ficha de objetivo del caso guía.
- [`02-polars-vs-pandas.md`](02-polars-vs-pandas.md) — cuándo cada uno.
- [`03-esquema-al-leer.md`](03-esquema-al-leer.md) — `schema_overrides` y por qué.
- [`04-limpieza-5-reglas.md`](04-limpieza-5-reglas.md) — las 5 funciones + HALLAZGOS.
- [`05-eda-manual-vs-auto.md`](05-eda-manual-vs-auto.md) — Tukey, correlaciones, skim, profiling.
- [`06-taller.md`](06-taller.md) — aplicación a dataset propio.
- [`07-troubleshooting.md`](07-troubleshooting.md) — errores reales que vimos.
- [`08-checklist-dia-de-clase.md`](08-checklist-dia-de-clase.md) — 15 minutos antes.

**Cuadernos del proyecto** (`projects/00-segmentacion-clientes/notebooks/`):
`00_objetivo_de_analisis`, `01_comprension_datos`, `02_limpieza`,
`03_variables_cliente`, `04_eda`, `05_eda_automatizado`.

**Módulos del paquete** (`src/segmentacion/`):
`data/cargar.py`, `data/limpieza.py`, `features/rfm.py`, `eda/resumen.py`.

**Scripts**: `descargar_datos.py` (SHA-256), `figuras_u2.py` (4 figuras
de referencia de la Unidad 2), `perfil.py` (HTML del perfil).

## 5. Quick start (macOS/Linux)

```bash
# desde la raíz del repositorio
uv sync --extra segmentacion-clientes --extra dev
uv run python projects/00-segmentacion-clientes/scripts/descargar_datos.py
cd projects/00-segmentacion-clientes
uv run jupyter lab notebooks/
```

Orden recomendado de los cuadernos:
`00 → 01 → 02 → 04 → 03 → 05`.

> Windows: ver [`07-troubleshooting.md`](07-troubleshooting.md) sección
> «ExecutionPolicy antes del primer script».

## 6. Autoverificación (sin spoiler)

1. ¿Por qué `pl.read_excel(..., schema_overrides={"Invoice": pl.Utf8})`
   es la **lección** del cuaderno 01 y no un detalle?
   _(ver [`03-esquema-al-leer.md`](03-esquema-al-leer.md) §1.)_
2. ¿Qué comprueba `tests/test_cifras_referencia.py` y qué garantiza
   que esté marcado `slow`?
   _(ver [`04-limpieza-5-reglas.md`](04-limpieza-5-reglas.md) §4.)_
3. ¿Cuál es la diferencia entre `recencia`, `recencia_fuga` y por qué
   `construir_rfm` **no** incluye la segunda?
   _(ver [`05-eda-manual-vs-auto.md`](05-eda-manual-vs-auto.md) §5.)_
4. Si el perfil HTML (`fg-data-profiling`) muestra una correlación
   Pearson distinta a la que calculaste a mano, ¿cuál crees?
   _(ver [`05-eda-manual-vs-auto.md`](05-eda-manual-vs-auto.md) §3.)_
5. ¿Qué fila de `HALLAZGOS.md` **no** se materializa en U2 sino en U3,
   y por qué?
   _(ver [`04-limpieza-5-reglas.md`](04-limpieza-5-reglas.md) §5.)_

## 7. Alternativas con criterio (fecha de referencia: 2026-10-19)

| Alternativa | Cuándo elegirla | Qué dejas en la mesa |
|---|---|---|
| **pandas** solo | Datasets < 100 MB; ecosistema sklearn ya pandas | Lectura con esquema estricto; lazy; velocidad |
| **DuckDB** + pandas | SQL cómodo; integración con parquet | Idioma del equipo (SQL vs. DataFrame API) |
| **PySpark** | Datasets > 10 GB; cluster disponible | Complejidad operativa; no justifica en 1 M filas |
| **Polars 2.x** (elegida) | Entre 100 MB y 10 GB; equipo sin cluster; lazy pedagógico | Comunidad de ejemplos todavía más chica |

## 8. Anti-patrones

| No hagas | Por qué | Qué hacer en su lugar |
|---|---|---|
| `pd.read_excel(ruta)` sin dtype | El motor adivina y pierde 19 494 cancelaciones silenciosamente | `pl.read_excel(ruta, schema_overrides={"Invoice": pl.Utf8, "StockCode": pl.Utf8, "Customer ID": pl.Int64})` |
| Imputar la mediana a `Customer ID` | Inventas realidad; cambias la cardinalidad de clientes | `exigir_cliente` + declarar la pérdida en `HALLAZGOS.md` |
| `!pip install polars` en una celda | Rompe el `uv.lock`; nada reproducible | Declararlo en `pyproject.toml`; `uv sync` |
| Guardar outputs del cuaderno en Git | Diffs ilegibles; CI roja; cuadernos pesan 30 MB | `nbstripout` como hook de pre-commit |
| Calcular Tukey "a ojo" sobre `Price` sin corte | 10 953,50 £ domina el gráfico y sesga la lectura | `log1p` **antes** del Tukey, o cerca inferior en escala original |

## 9. Cierre y puente

- **Tarea:** terminar el taller ([`06-taller.md`](06-taller.md)) y actualizar
  `HALLAZGOS.md` del dataset elegido.
- **Puente a S03:** _«Las correlaciones que calculamos describen el
  pasado. ¿Qué pasa si el pasado incluye información del futuro — y no
  lo notamos? La próxima sesión demuestra cómo esa fuga eleva el AUC
  de 0,795 a 0,955 sin cambiar el modelo.»_

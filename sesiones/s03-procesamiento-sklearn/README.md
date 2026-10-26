---
title: "S03 · Procesamiento, pipelines, modelos"
parent: Sesiones
nav_order: 3
has_children: true
---
# Sesión S03 · Procesamiento, pipelines, modelos y LLM — Guía para el instructor

> Revisión: **2026-10-19** · Versiones verificadas: Python 3.11.14, Polars 2.0.0, scikit-learn 1.9.1, Pydantic 2.11 · Spec que implementa: [`docs/specs/SPEC_procesamiento_U3.md`](../../docs/specs/SPEC_procesamiento_U3.md) + [`SPEC_llm_transformador.md`](../../docs/specs/SPEC_llm_transformador.md) · Proyecto que soporta: [`projects/00-segmentacion-clientes/`](../../projects/00-segmentacion-clientes/).

**No asumas que un contenido sin fecha sigue siendo correcto.** La pregunta
que organiza esta sesión es:

> _«¿Qué garantiza que el modelo que cuelgo hoy en producción no vio
> ninguna de las decisiones que lo van a evaluar mañana?»_ — porque esa
> garantía es técnica (código) y declarativa (metadatos), no de confianza.

---

## 0. Para antes de clase

- [ ] Revisar [`08-checklist-dia-de-clase.md`](08-checklist-dia-de-clase.md).
- [ ] `uv sync --extra segmentacion-clientes --extra dev`.
- [ ] `uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_recompra.py`
      verde (20/20).
- [ ] Si vas a enseñar el bloque LLM: `.env` con las tres variables
      (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODELO`) o decidir que la
      clase corre con `FakeLLM` (recomendado).

## 1. Objetivos de aprendizaje

Al terminar la sesión el estudiante puede:

1. **Diseñar** cohortes de entrenamiento (A) y prueba (B) por corte
   temporal, con etiqueta de recompra a horizonte de 90 días.
2. **Identificar** dos tipos de fuga temporal (en la variable y en el
   preprocesamiento) y demostrar su efecto sobre el AUC.
3. **Construir** un `ColumnTransformer` con `log1p`, `StandardScaler`
   y `OneHotEncoder(handle_unknown="ignore")`, respetando la regla
   «`fit` después de la partición».
4. **Elegir** entre regresión logística, árbol y bosque mirando AUC,
   AP, F1 y una regla por recencia como línea base.
5. **Fijar** el umbral de decisión con el equipo de negocio (tabla de
   umbrales).
6. **Serializar** un modelo con ficha (`metadatos.json`) que haga
   `cargar()` fallar si cambia la versión de scikit-learn.
7. **Integrar** un LLM como un transformador más (12 categorías cerradas,
   `FakeLLM`, caché, comparación contra una regla por palabras clave).

## 2. Organización temporal (6 h de interacción · 12 h de trabajo independiente)

| Bloque | Duración | Contenido | Documento |
|---|---|---|---|
| **El problema** | 15 min | AUC 0,955 que es mentira | este README §3 |
| **A — Cohortes y fuga** | 60 min | `cohorte(ventas, cancelaciones, corte, horizonte)`; dos fugas | [`01-cohortes-y-fuga.md`](01-cohortes-y-fuga.md) |
| **B — Pipeline sin fuga** | 60 min | `NUM_LOG`, `NUM_STD`, `CAT`; `test_sin_fuga` | [`02-pipelines-sin-fuga.md`](02-pipelines-sin-fuga.md) |
| **C — Modelos y umbral** | 90 min | Logreg / árbol / bosque / regla; tabla de umbrales; curva de profundidad | [`03-modelos-y-umbral.md`](03-modelos-y-umbral.md) |
| **D — Artefacto y registro** | 45 min | `models/recompra_v1/metadatos.json`; guard de versión | [`04-artefacto-y-registro.md`](04-artefacto-y-registro.md) |
| **E — LLM como transformador** (opcional) | 60 min | 12 categorías cerradas; FakeLLM; caché; comparación vs. regla | [`05-llm-transformador.md`](05-llm-transformador.md) |
| **Taller** | 90 min | Aplicación a dataset propio | [`06-taller.md`](06-taller.md) |

## 3. El problema (antes de las herramientas)

En voz alta:

1. _«Si entrenamos la regresión logística con la variable `recencia_fuga`
   (que mezcla datos del pasado con del futuro), el AUC en la cohorte B
   salta de **0,795** a **0,955**. ¿Es un mejor modelo o un peor
   programador?»_

2. _«Si ajustamos el `StandardScaler` sobre A ∪ B porque "tenemos los
   datos", la media de `recencia` cambia de **168,7** a **186,9** días.
   El AUC apenas se mueve. ¿Es un problema?»_
   > Sí, pero sutil. El modelo se calibra contra información del futuro;
   > el día de inferencia, la calibración no coincide. Es la razón de
   > la regla «`fit` después de la partición».

3. _«Si el bosque entrenado hoy devuelve `AUC = 0,785` y mañana devuelve
   `AUC = 0,787`, ¿me importa? ¿O lo descarto como ruido?»_
   > Depende del contrato. Las métricas con `n_jobs=-1` fluctúan a
   > nivel ~2e-16 por el orden de hilos. Para el día del video, la
   > tolerancia aceptada es `1e-3` para AUC y `5e-3` para métricas
   > derivadas al umbral 0,5 (precision/recall/F1).

Las tres preguntas motivan los bloques A, B y D.

## 4. Materiales y artefactos

- [`01-cohortes-y-fuga.md`](01-cohortes-y-fuga.md) — `cohorte(...)` y las dos fugas.
- [`02-pipelines-sin-fuga.md`](02-pipelines-sin-fuga.md) — `construir_preprocesador` + `test_sin_fuga`.
- [`03-modelos-y-umbral.md`](03-modelos-y-umbral.md) — 3 modelos + regla + tabla de umbrales.
- [`04-artefacto-y-registro.md`](04-artefacto-y-registro.md) — `guardar()` / `cargar()` + Tabla 7.
- [`05-llm-transformador.md`](05-llm-transformador.md) — `texto_llm`, `llm_clientes`, prompt, caché.
- [`06-taller.md`](06-taller.md) — aplicación a dataset propio.
- [`07-troubleshooting.md`](07-troubleshooting.md) — errores reales ya vistos.
- [`08-checklist-dia-de-clase.md`](08-checklist-dia-de-clase.md).

**Cuadernos del proyecto** (`projects/00-segmentacion-clientes/notebooks/`):
`06_particion_y_fuga`, `07_ingenieria_caracteristicas`,
`08_llm_como_transformador` (opcional), `09_modelos_y_pipelines`,
`10_segmentacion_no_supervisada`.

**Módulos del paquete** (`src/segmentacion/`):
`features/{devoluciones, etiquetas, cohortes, texto_llm, llm_clientes}.py`,
`pipelines/preprocesamiento.py`, `models/{entrenar, registro}.py`,
`evaluation/{metricas, umbral}.py`.

**Scripts**: `caso_recompra.py` (envoltorio), `figuras_u3.py` (4 figuras del OVA),
`generar_gold.py` (fixture para evaluar el LLM).

## 5. Quick start (macOS/Linux)

```bash
# desde la raíz del repositorio
uv sync --extra segmentacion-clientes --extra dev
cd projects/00-segmentacion-clientes
# Si vienes de S02 ya está hecho; si no:
uv run python scripts/descargar_datos.py
# Caso de referencia:
uv run python scripts/caso_recompra.py
uv run python scripts/figuras_u3.py
# Cuadernos:
uv run jupyter lab notebooks/
```

> Windows: ver [`07-troubleshooting.md`](07-troubleshooting.md).

## 6. Autoverificación

1. ¿Por qué `cohorte` **termina con** `.sort("Customer ID")`? ¿Qué
   rompe si no está?
   _(ver [`01-cohortes-y-fuga.md`](01-cohortes-y-fuga.md) §4.)_
2. ¿Qué comprueba `tests/pipelines/test_sin_fuga.py` y por qué usa dos
   cohortes con medias de `recencia` muy distintas?
   _(ver [`02-pipelines-sin-fuga.md`](02-pipelines-sin-fuga.md) §3.)_
3. ¿Por qué 0,3 es un umbral razonable para el equipo de marketing y
   0,5 no?
   _(ver [`03-modelos-y-umbral.md`](03-modelos-y-umbral.md) §4.)_
4. ¿Qué pasa si `metadatos.json` dice `sklearn: "1.9.0"` y el entorno
   tiene `1.9.1`?
   _(ver [`04-artefacto-y-registro.md`](04-artefacto-y-registro.md) §3.)_
5. ¿Por qué `FakeLLM` es el cliente por defecto en el cuaderno 08 y no
   el cliente real?
   _(ver [`05-llm-transformador.md`](05-llm-transformador.md) §2.)_

## 7. Alternativas con criterio (2026-10-19)

| Alternativa | Cuándo elegirla | Qué dejas en la mesa |
|---|---|---|
| **sklearn Pipeline** (elegida) | Flujos de ≤ 20 transformaciones; buena interoperabilidad | Metaflow, Prefect (orquestación real); DAG explícito |
| **TPOT / AutoML** | Prototipo rápido; equipo sin tiempo | Control del pipeline; auditabilidad |
| **Spark MLlib** | Datasets > 50 GB o inferencia distribuida | Verbosidad; difícil de debuggear |
| **Pipelines caseros con joblib** | Un modelo único sin cadena de transformaciones | Separar `fit` y `transform`; serialización estándar |

## 8. Anti-patrones

| No hagas | Por qué | Qué hacer en su lugar |
|---|---|---|
| Calcular variables antes de partir | Fuga silenciosa | Calcular después, con un `pipeline.fit(X_train, y_train)` que incluya transformadores |
| `fit_transform(A ∪ B)` | Ya viste B al calibrar | `fit(A)` → `transform(B)` |
| `random_state=None` por "ciencia" | No reproducible → no se puede comparar | `random_state=SEMILLA` siempre |
| Serializar solo `pipeline.joblib` | U4 no sabe qué columnas espera | `metadatos.json` con las 10 campos de la Tabla 7 |
| Nombrar al proveedor del LLM | Rompe el `grep -r` de la auditoría | Solo el adaptador y el README nombran proveedores |

## 9. Cierre y puente

- **Tarea:** terminar [`06-taller.md`](06-taller.md) y producir
  `models/<tu_proyecto>_v1/metadatos.json`.
- **Puente a S04:** _«Tenemos un modelo serializado y una ficha. ¿Cómo
  monta esto una app que un usuario de negocio pueda abrir y usar? La
  próxima sesión conecta el artefacto con Streamlit y define el
  contrato de inferencia que la Unidad 4 implementa.»_

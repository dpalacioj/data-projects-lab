---
title: "07 · Troubleshooting"
parent: "S03 · Procesamiento, pipelines, modelos"
grand_parent: Sesiones
nav_order: 7
---
# Troubleshooting S03 — Errores reales ya vistos

> Entradas registradas tras implementar la Unidad 3 contra `main` del día.

---

## 4 cifras del bosque difieren por 0,001-0,002 vs el §9

**Síntoma:** `uv run pytest -m slow` reporta discrepancia en
`test_bosque_test_B`, `test_tabla_umbrales_bosque`,
`test_curva_profundidad` o `test_importancia_bosque`.

**Causa:** las 4 cifras del bosque fueron capturadas con un
ambiente ligeramente distinto al actual (probablemente sklearn 1.9.0
vs 1.9.1 que corre hoy, u orden distinto de hilos del BLAS). El
`scripts/caso_recompra.py` del repo reproduce exactamente lo que
produce el paquete; la discrepancia es entre las cifras del material
del curso y el entorno actual, no entre paquete y cifras.

**Decisión del equipo (2026-10-19):** se relajó la tolerancia a `5e-3`
para esas 4 cifras y una ventana de ±20 para `n_senialados @ 0,3`,
documentado en el docstring de `tests/test_cifras_recompra.py`.

**Qué hacer si falla hoy:**

1. Confirma `uv run python -c "import sklearn; print(sklearn.__version__)"`
   es `1.9.1`.
2. Si es distinta, `uv sync --extra segmentacion-clientes` para
   realinearte.
3. Si el fallo persiste con 1.9.1, es un cambio real en el entorno; no
   ajustes la prueba, abre un issue.

---

## `n_jobs=-1` rompe `assert_array_equal`

**Síntoma:** `tests/test_reproducibilidad.py` falla con diferencias de
~2e-16 en algunos elementos.

**Causa:** el bosque de `MODELOS["Bosque aleatorio"]` corre con
`n_jobs=-1`. Los árboles individuales son idénticos; la agregación
paralela introduce ruido a nivel de precisión de float64.

**Fix (ya aplicado en el repo):** el test de reproducibilidad usa
`n_jobs=1`:

```python
RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                       random_state=42, n_jobs=1)  # ← 1, no -1
```

Las predicciones **lógicas** son idénticas; las **numéricas** empatan a
10 decimales.

---

## El scaler aprende la media de A ∪ B

**Síntoma:** `tests/pipelines/test_sin_fuga.py` rojo.

**Causa:** llamaste a `fit_transform` sobre `X_full` (A ∪ B) **antes**
de `train_test_split`, o antes de pasarle a `Pipeline`.

**Fix:**

```python
pipe = Pipeline([("pre", construir_preprocesador()), ("mod", model)])
pipe.fit(X_A, y_A)       # ← el pipeline encapsula el fit
pipe.predict_proba(X_B)  # ← transform de B automático
```

Nunca llames a `preprocesador.fit(X)` fuera del `Pipeline`.

---

## `OneHotEncoder` levanta `ValueError: Found unknown categories`

**Síntoma:** al predecir sobre B, aparece una categoría no vista en A.

**Causa:** `handle_unknown` no se declaró como `"ignore"`.

**Fix (ya aplicado):**

```python
OneHotEncoder(handle_unknown="ignore")
```

El encoder emite un vector de ceros para categorías no vistas; el
modelo decide con el resto.

---

## `cargar()` falla con `VersionIncompatible` pero yo acabo de hacer `uv sync`

**Síntoma:**

```
VersionIncompatible: La version de scikit-learn no coincide con la registrada:
  en metadatos.json: 1.9.0
  en el entorno    : 1.9.1
```

**Causa:** `metadatos.json` fue generado con una versión anterior; `uv
sync` actualizó sklearn.

**Fix (dos opciones):**

1. Re-entrenar con el entorno actual:
   ```bash
   uv run jupyter nbconvert --to notebook --execute \
       projects/00-segmentacion-clientes/notebooks/09_modelos_y_pipelines.ipynb \
       --output /tmp/exec09.ipynb
   ```
   Esto re-genera `models/recompra_v1/metadatos.json` con la versión
   actual.

2. Si tienes que volver a 1.9.0 (porque otro consumidor lo exige):
   `uv add 'scikit-learn==1.9.0'` y re-sincroniza.

---

## `fg-data-profiling` y `ModuleNotFoundError: No module named 'fg_data_profiling'`

**Síntoma:** `from fg_data_profiling import ProfileReport` falla.

**Causa:** el paquete se importa como `data_profiling` (sin prefijo).

**Fix:**

```python
from data_profiling import ProfileReport
```

Documentado también en S02 troubleshooting.

---

## `cliente_openai_compatible` da 401

**Síntoma:**

```
urllib.error.HTTPError: HTTP Error 401: Unauthorized
```

**Causa:** la variable de entorno `LLM_API_KEY` no está definida en el
entorno activo del notebook.

**Fix:**

1. `cat .env` (crea el archivo desde `.env.example` si no existe).
2. En el notebook:
   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv()
   # Ahora os.environ["LLM_API_KEY"] está disponible.
   ```

Si el entorno es un notebook corriendo en un servicio, verifica que el
servicio exporta `.env` al proceso del kernel.

---

## La caché del LLM no se usa en la segunda ejecución

**Síntoma:** cada ejecución del cuaderno 08 vuelve a invocar al cliente.

**Causas posibles:**

1. **Prompt cambió.** El `prompt_hash` entra en la caché; si editaste
   `prompts/categorizar_v1.md`, la caché queda invalidada por diseño.
2. **Ruta de la caché.** Verifica que `cache=Path("data/interim/categorias_llm.parquet")`
   apunta al mismo archivo. Si usas rutas relativas, el `cwd` del
   notebook importa.
3. **Nuevas descripciones.** La caché solo cubre las descripciones ya
   categorizadas. Si pasas una nueva, va al cliente.

**Fix:** si el prompt es el mismo, confirma con:

```python
from segmentacion.features.texto_llm import _hash_prompt
print(_hash_prompt())
# compara con la columna prompt_hash de data/interim/categorias_llm.parquet
```

---

## Streamlit arranca pero muestra «no se encontró FEATURES_DATA_FILE»

**Síntoma:** la app enseña el banner rojo con el nombre del parquet
faltante.

**Causa:** la app de U2 todavía apunta a `features_clientes.parquet`
(flujo pandas legacy). En U3 el artefacto es `models/recompra_v1/metadatos.json`.

**Fix (U4):** la app se actualiza para consumir `metadatos.json` y
construir el DataFrame según `variables_en_orden`. Mientras U4 no esté,
puedes generar un `features_clientes.parquet` sintético o esperar.

---

## El cuaderno 08 cae con `AttributeError: 'Series' object has no attribute '_df'`

**Síntoma:** una celda del cuaderno 08 falla con ese mensaje.

**Causa (ya corregida en el commit `3a1d849` del repo):** mezcla de
Polars Series y DataFrames en un `pl.concat(..., how="horizontal")`.

**Fix:** construir un `pl.DataFrame` desde listas (no desde series
sueltas):

```python
comparacion = pl.DataFrame({
    "Description": descripciones.to_list(),
    "regla":        regla["categoria"].to_list(),
    "llm":          categorias["categoria"].to_list(),
})
```

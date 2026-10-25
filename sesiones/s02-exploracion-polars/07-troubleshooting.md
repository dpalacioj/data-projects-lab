# Troubleshooting S02 — Errores reales ya vistos

> Entradas registradas tras implementar la Unidad 2 contra `main` del día. Si tu error no está aquí, abre un issue con el stack completo.

---

## `fastexcel` no se instala / `pl.read_excel` es lentísimo

**Síntoma:** `pl.read_excel` tarda 2-3 minutos sobre el `.xlsx` de 45 MB.

**Causa:** el motor por defecto cambió entre versiones de Polars. Hoy
`fastexcel` es el motor correcto para 2.x.

**Fix:**

```bash
uv sync --extra segmentacion-clientes   # ya incluye fastexcel>=0.11
```

Si sigue lento, pasa a Parquet una vez:

```bash
uv run python projects/00-segmentacion-clientes/scripts/descargar_datos.py
# A partir de aquí, siempre leer desde Parquet.
```

---

## `ModuleNotFoundError: No module named 'fg_data_profiling'`

**Síntoma:** `from fg_data_profiling import ProfileReport` falla.

**Causa:** `fg-data-profiling` es un fork de `ydata-profiling` y el
paquete **se importa como `data_profiling`** (sin prefijo `fg_`).

**Fix:**

```python
from data_profiling import ProfileReport
```

Está documentado en `src/segmentacion/features/texto_llm.py` también y
es un gotcha conocido.

---

## Las cifras no coinciden con la spec

**Síntoma:** `uv run pytest -m slow` rojo. Alguno de los conteos de §4
está fuera (tolerancia 0).

**Causa posible:**

1. **Hash del zip cambió.** UCI re-publicó el archivo. Verifica con
   `sha256sum data/raw/online_retail_II.zip` contra la constante
   `SHA256_ZIP` en `scripts/descargar_datos.py`. Si no coincide, el
   archivo es nuevo; la spec está desactualizada.
2. **Versión de Polars.** La spec se escribió con Polars 2.0.0. Un
   bump mayor (3.x) podría cambiar la semántica de `unique()` o
   `n_unique()`.
3. **Python** 3.11 vs 3.13: no debería afectar las cifras del §4
   (son conteos deterministas).

**Qué hacer:** reporta la discrepancia en el PR; no ajustes la prueba.
Decide con el instructor si se actualiza la spec y el OVA, o si se
revierte el entorno.

---

## `preparar_ventas` no muestra el log

**Síntoma:** no sale la línea `[HH:MM:SS] INFO ... pipeline de limpieza: ...`.

**Causa:** el nivel del logger está en `WARNING`.

**Fix:**

```python
import logging
logging.getLogger("segmentacion.data.limpieza").setLevel(logging.INFO)
```

O ejecuta en un cuaderno Jupyter; el logger está configurado para
imprimir en stdout.

---

## El perfil HTML pesa > 10 MB

**Síntoma:** `perfil_muestra.html` ronda los 20 MB. Casi no abre.

**Causa:** `minimal=False` o muestra demasiado grande.

**Fix:**

```python
ProfileReport(muestra, minimal=True, progress_bar=False).to_file(destino)
```

Con `minimal=True` y 100 000 filas el HTML pesa ~2 MB.

---

## Mi criterio de negocio no se puede medir

**Síntoma:** al escribir la Tabla 2 del cuaderno 00, no sabes qué poner
en «Criterio de éxito de negocio».

**Fix (procedimiento):**

1. Pregúntate: _¿quién actuará sobre el modelo?_ (marketing,
   operaciones, finanzas).
2. Pregúntale (o imagina) qué número verá cada semana al revisar el
   modelo. Esa es tu métrica.
3. Si no hay número semanal, es porque todavía no hay decisión
   recurrente. En ese caso, el modelo no vale la pena.

---

## Windows: `ExecutionPolicy` antes del primer script

**Síntoma:** PowerShell rehúsa ejecutar `.ps1` o scripts dentro de
Jupyter.

**Fix (como administrador, una sola vez):**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego:

```powershell
.venv\Scripts\Activate.ps1
```

---

## `uv sync` descarga 2 GB

**Síntoma:** la primera sincronización es gigante.

**Causa:** los extras `web`, `experiment` y `all` incluyen MLflow,
Gradio, Streamlit, etc.

**Fix:** sincroniza solo lo que necesitas:

```bash
uv sync --extra segmentacion-clientes --extra dev
```

Para U3 sumas `--extra` del siguiente proyecto; no hace falta `--all-extras`.

---

## Zonas horarias y `pl.Datetime`

**Síntoma:** al convertir `InvoiceDate` a pandas, los timestamps se
corren una hora.

**Causa:** Polars 2.x guarda `Datetime` con `time_zone=None` (naive);
pandas asume UTC en algunas conversiones.

**Fix:** si necesitas pandas, convierte con `.to_pandas()` y luego
`pd.to_datetime(..., utc=False)` explícito. Para U2 trabajas en Polars
y no deberías tener este problema.

---

## Mi `nbstripout` borra las imágenes de salida

**Síntoma:** las gráficas desaparecen del `.ipynb` al hacer commit.

**Causa:** es el comportamiento esperado. El hook de pre-commit elimina
outputs para que los diffs sean legibles.

**Qué hacer:** reconstruye con `uv run jupyter nbconvert --execute`
cuando necesites ver outputs. Para figuras persistentes, usa
`scripts/figuras_u2.py` que las deja en `reports/figures/`.

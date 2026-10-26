---
nav_exclude: true
---
# Guía de Lectura: Segmentación de Clientes

> Guía ordenada por **Unidad del módulo**. Dedica el tiempo sugerido a
> cada cuaderno y responde las preguntas de reflexión antes de pasar al
> siguiente.

## Objetivo

Aprender, con un caso real, el flujo completo de un proyecto de
**clustering no supervisado**: desde el objetivo de análisis hasta una
demo en Streamlit, pasando por lectura y limpieza con Polars,
construcción de variables, EDA (manual y automatizado), pipelines,
K-Means, DBSCAN e interpretación de segmentos.

- **Dataset:** Online Retail II (UCI 502, ~1 M transacciones,
  ~5 943 clientes). Procedencia y licencia en
  [`data/README.md`](data/README.md).
- **Stack U2:** Polars 2.x para todo el flujo; pandas solo en el perfil
  HTML (cuaderno 05) sobre una muestra.
- **Stack U3 (pendiente):** scikit-learn >= 1.9, pipelines, clustering.
- **Demo:** Streamlit interactivo (reemplaza su insumo con la salida
  de U3 cuando esté lista).

---

## Unidad 2 (00-05) — RAE 1.1 · ~4 horas

Orden recomendado: **00 → 01 → 02 → 04 → 03 → 05**. El EDA (04) usa el
Parquet de ventas que deja el 02; el 03 (RFM) depende del 02 pero no
del 04, por eso se puede explorar antes del RFM.

### 0. Objetivo de análisis (15 min)

```
notebooks/00_objetivo_de_analisis.ipynb
```

**Qué aprenderás:**
- Qué es la fase 1 de CRISP-DM y por qué se escribe antes de tocar código.
- Diferencia entre objetivo de negocio, objetivo de análisis y criterio
  de éxito.
- Qué entregables deja la U2 y qué recoge la U3.

---

### 1. Comprensión de datos (30 min)

```
notebooks/01_comprension_datos.ipynb
```

**Qué aprenderás:**
- Lectura con `polars.read_parquet` y esquema declarado al leer.
- Diagnóstico de nulos, duplicados, cancelaciones y rangos sospechosos
  — Tabla 4 del OVA reproducida.

---

### 2. Limpieza (40 min)

```
notebooks/02_limpieza.ipynb
```

**Qué aprenderás:**
- Reglas de limpieza como funciones (`quitar_duplicados`,
  `separar_cancelaciones`, `exigir_cliente`, `filtrar_validas`).
- `preparar_ventas` encadena con `.pipe` y registra `n_antes/n_despues`.
- Secuencia esperada: 1 067 371 → 779 425 (73 %).
- Deja `data/processed/{ventas_validas,cancelaciones}.parquet`.

**Código clave:** `src/segmentacion/data/limpieza.py`,
`src/segmentacion/data/cargar.py`.

---

### 3. EDA manual (45 min)

```
notebooks/04_eda.ipynb
```

**Qué aprenderás:**
- Cuartiles y cercas de Tukey (`Price = 7,50`; `Quantity = 27`).
- Diferencia entre Pearson y Spearman con un ejemplo real
  (−0,005 vs −0,428).
- Spearman frecuencia-monetario = 0,858; 20 % de clientes = 77,2 %
  del total.

**Código clave:** `src/segmentacion/eda/resumen.py`.

---

### 4. Variables por cliente — RFM (30 min)

```
notebooks/03_variables_cliente.ipynb
```

**Qué aprenderás:**
- Firma `construir_rfm(ventas, fecha_corte)`.
- `FECHA_SNAPSHOT = max(InvoiceDate) + 1 día`.
- Por qué el filtro `InvoiceDate < fecha_corte` es importante para U3.
- Deja `data/processed/rfm_u2.parquet` — insumo de U3.

**Código clave:** `src/segmentacion/features/rfm.py`.

---

### 5. EDA automatizado (30 min)

```
notebooks/05_eda_automatizado.ipynb
```

**Qué aprenderás:**
- `skim` sobre Polars (sin convertir).
- Muestra reproducible con `seed=42` → `fg-data-profiling`.
- HTML en `reports/perfil_muestra.html` (gitignored).

---

## Unidad 3 (06-10) — pendiente, próxima SPEC

Los cuadernos 06-10 nacen de `SPEC_procesamiento_U3.md` y reutilizan,
sin cambios de firma, el paquete `segmentacion` y los Parquet que deja
U2. Mientras llegan, los cuadernos actuales en `notebooks/legacy/` están
**en migración**:

```
notebooks/legacy/05_preprocessing_pipelines.ipynb  →  07_ingenieria_caracteristicas
notebooks/legacy/06_clustering_kmeans.ipynb        →  10_segmentacion_no_supervisada
notebooks/legacy/07_clustering_dbscan.ipynb        →  10_segmentacion_no_supervisada
notebooks/legacy/08_model_comparison_and_interpretation.ipynb → 10_segmentacion_no_supervisada
```

Si quieres ejecutarlos hoy, siguen funcionando: ya se actualizaron sus
`import` tras renombrar el paquete. Pero están congelados a la lógica
pandas original; U3 los reemplaza.

---

## Pruebas y scripts

```bash
uv run pytest                                       # rápido (test_limpieza, test_rfm)
uv run pytest -m slow                               # cifras de §4 del OVA
uv run python scripts/descargar_datos.py            # descarga + hash
uv run python scripts/figuras_u2.py <parquet> <out> # Figuras 6-9
uv run python scripts/perfil.py                     # HTML de perfil
```

---

## Resumen de archivos clave

| Archivo | Propósito |
|---|---|
| `src/segmentacion/config.py` | Rutas y constantes. |
| `src/segmentacion/data/cargar.py` | `ESQUEMA`, `leer_excel_con_esquema`, `leer_parquet`. |
| `src/segmentacion/data/limpieza.py` | 5 reglas + `preparar_ventas`. |
| `src/segmentacion/data/loader.py` | Loader pandas (lo usan los legacy). |
| `src/segmentacion/eda/resumen.py` | `cuartiles`, `cercas_tukey`, `asociaciones`, `ventas_por_mes`, `matriz_dia_hora`. |
| `src/segmentacion/features/rfm.py` | `construir_rfm(ventas, fecha_corte)`. |
| `src/segmentacion/features/preprocessing.py` | Pipelines sklearn (legacy / U3). |
| `src/segmentacion/models/{clustering,profiling}.py` | Clustering y perfiles (legacy / U3). |
| `tests/test_limpieza.py` / `test_rfm.py` | Rápidos, con filas inventadas. |
| `tests/test_cifras_referencia.py` | `@pytest.mark.slow`, cifras del OVA. |
| `scripts/figuras_u2.py` | Figuras 6-9 del OVA (reproducibles). |
| `scripts/perfil.py` | HTML del perfil (CI / sin Jupyter). |
| `HALLAZGOS.md` | Hallazgos de U2 → insumo de U3. |

---

## Tips para maximizar tu aprendizaje

1. **Ejecuta cada celda y juega con los parámetros.** Cambia `seed`,
   cambia `k`, cambia `eps`. Observa qué pasa.
2. **Responde las preguntas de reflexión** al final de cada notebook
   antes de continuar al siguiente.
3. **Discute con un compañero.** Las decisiones de negocio rara vez
   son obvias.
4. **Toma notas en un cuaderno aparte.** Anota las decisiones que
   tomarías tú diferente.

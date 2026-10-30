# SPEC — `projects/00-segmentacion-clientes` como recurso de la Unidad 2 (Polars)

**Repositorio:** `dpalacioj/data-projects-lab` · **Carpeta:** `projects/00-segmentacion-clientes/`
**Propósito:** convertir el proyecto existente en el recurso de referencia de la Unidad 2 (RAE1.1), de modo que el videotutorial, el material del curso y el taller apunten al mismo código y a las mismas cifras.
**Estado:** especificación **ejecutada**. Las referencias a `notebooks/legacy/` describen el estado intermedio de la migración; la carpeta ya fue eliminada tras consolidar los cuadernos de clustering en `10_segmentacion_no_supervisada.ipynb`.
**Fecha límite sugerida:** antes de grabar el videotutorial (la producción de insumos pasa a Competencias y Medios Digitales el 11 de diciembre de 2026).
**Orden de implementación:** **1 de 3.** Esta spec va primero. Después vienen `SPEC_procesamiento_U3.md` (2 de 3, que incluye `SPEC_llm_transformador.md`) y `SPEC_material_nicolas.md` (3 de 3). La U3 depende de lo que esta spec deja: `src/segmentacion/data/`, el Parquet, `cancelaciones.parquet` y `HALLAZGOS.md`.
**Ubicación en el repositorio:** las specs viven en `docs/specs/` (raíz del repo). Los scripts de referencia (`figuras_u2.py`, y luego `caso_recompra.py` y `figuras_u3.py`) se copian a `projects/00-segmentacion-clientes/scripts/`.

---

## 1. Contexto y problema

El proyecto hoy sigue la secuencia comprensión → limpieza → features → EDA → pipelines → clustering → Streamlit, en **pandas**, con un `src/` modular, `GUIA_LECTURA.md` y `DICCIONARIO_DATOS.md`. Es la mejor base del repositorio para la Unidad 2, pero el microcurrículo exige «preparación y exploración de datos con Polars» y «herramientas para automatizar los EDA», y el material de la unidad ya está escrito con cifras y funciones concretas (ver §4). El proyecto debe alinearse con eso sin perder lo que ya enseña bien (decisiones justificadas, preguntas de reflexión, código en `src/`).

Vínculo con las guías entregadas:

| Documento de la unidad | Qué espera del repositorio |
|---|---|
| Material del curso, tema 2.1 (Tabla 3, Figuras 3 y 4, bloques de código) | Lectura con `schema_overrides`, Parquet en `data/raw/`, funciones `quitar_duplicados`, `separar_cancelaciones`, `exigir_cliente`, `filtrar_validas`, `preparar_ventas` encadenadas con `pipe` |
| Material del curso, tema 2.2 (Figuras 6–9, Tablas 4, 5 y 7) | Cuartiles, ventas por mes, mapa día×hora, cercas de Tukey, Pearson vs. Spearman, agregación por cliente, `HALLAZGOS.md` |
| Material del curso, tema 3 (Tabla 6) | `skim` sobre Polars y perfil HTML con `fg-data-profiling` sobre una muestra, guardado en `reports/` |
| Videotutorial (guion minutado, bloques 1–6) | Cuadernos 00–05 ejecutables en orden; `tests/test_limpieza.py`; `HALLAZGOS.md` abierto para editar en vivo |
| Taller evaluativo | Estructura que el estudiante pueda imitar: README con objetivo/criterio, `src/`, `tests/`, `HALLAZGOS.md` |

## 2. Objetivos

1. Portar la comprensión, limpieza, construcción de variables y EDA (cuadernos 01–04) de pandas a **Polars 2.x**, manteniendo pandas solo donde una herramienta lo exija (perfil HTML).
2. Añadir un cuaderno inicial de **objetivo de análisis** (CRISP-DM fase 1) y uno final de **EDA automatizado**.
3. Dejar las **cifras de referencia reproducibles** desde un script, con una prueba que las verifique.
4. Separar con claridad lo que pertenece a la Unidad 2 (00–05) de lo que pertenece a la Unidad 3 (pipelines, clustering, app), sin borrar esto último.

## 3. Fuera de alcance

- Cambiar la lógica de los cuadernos de clustering (05–08 actuales) o de la app de Streamlit (Unidad 3/4; ver `SPEC_procesamiento_U3.md`). **Sí** entra en alcance actualizar sus `import` para que sigan funcionando tras renombrar el paquete (§5.4).
- Reescribir `clasificador-meli`, `titanic` o `prediccion-precios-automoviles`.
- Publicar el HTML del perfil en el repositorio (se genera; no se versiona).

## 4. Cifras de referencia que el código debe reproducir

Calculadas el 16 de octubre de 2026 con Polars 2.0.0 sobre `online_retail_II.xlsx` (UCI id 502, CC BY 4.0). Son las cifras de referencia; si cambian al re-descargar, se actualizan **las dos cosas**: la spec y la prueba.

| Métrica | Valor |
|---|---|
| Filas totales | 1 067 371 |
| Filas sin `Customer ID` | 243 007 (22,77 %) |
| `Description` nulas (todas con `Price` = 0 y sin cliente) | 4 382 |
| Duplicados exactos | 34 335 |
| Facturas que empiezan por `C` | 19 494 |
| Facturas no numéricas (C + A) | 19 500 |
| `Quantity` < 0 sin factura C | 3 457 |
| `Price` = 0 / `Price` < 0 | 6 202 / 5 |
| Filas con `StockCode` no estándar (no `^\d{5}`) | 6 093 |
| Países / clientes únicos / productos / facturas | 43 / 5 943 / 5 305 / 53 628 |
| Pipeline secuencial: tras `unique` → tras quitar C → tras exigir cliente → tras `Quantity>0 & Price>0` | 1 033 036 → 1 013 932 → 779 495 → **779 425** (73,0 %) |
| Ventas válidas: `Price` mediana / Q3 / media / máx | 1,95 / 3,75 / 3,22 / 10 953,50 |
| Ventas válidas: `Quantity` mediana / máx | 6 / 80 995 |
| Cerca de Tukey `Price` / % que la supera | 7,50 / 8,40 % |
| Cerca de Tukey `Quantity` / % que la supera | 27 / 6,56 % |
| Pearson / Spearman `Quantity`–`Price` | −0,005 / −0,428 |
| Spearman frecuencia–monetario (por cliente) | 0,858 |
| Participación del 20 % de clientes de mayor valor | 77,2 % |
| Filas en sábado (ventas válidas) | 400 |
| Rango de fechas | 2009-12-01 07:45 → 2011-12-09 12:50 |

## 5. Diseño

### 5.1 Estructura objetivo

```
projects/00-segmentacion-clientes/
├── README.md                     # + sección "Objetivo de negocio, objetivo de análisis, criterio de éxito" (Tabla 2 de la spec)
├── HALLAZGOS.md                  # NUEVO · tabla hallazgo | evidencia | decisión (Tabla 7 de la spec)
├── GUIA_LECTURA.md               # actualizar orden y tiempos; marcar 00–05 = Unidad 2, 06+ = Unidad 3
├── DICCIONARIO_DATOS.md          # añadir columna "tipo declarado en la lectura"
├── data/
│   ├── raw/                      # online_retail_II.xlsx + .parquet (gitignored; script de descarga)
│   └── README.md                 # procedencia, licencia CC BY 4.0, fecha de descarga, hash SHA-256
├── notebooks/
│   ├── 00_objetivo_de_analisis.ipynb      # NUEVO · CRISP-DM fase 1 (plantilla de ficha + caso)
│   ├── 01_comprension_datos.ipynb         # PORTAR · lectura con esquema, diagnóstico (Tabla 4)
│   ├── 02_limpieza.ipynb                  # PORTAR · llama a src; conteo de filas por regla (Figura 4)
│   ├── 03_variables_cliente.ipynb         # PORTAR · RFM en Polars (puente a U3; se mantiene aquí)
│   ├── 04_eda.ipynb                       # PORTAR · univariado, temporal, anomalías, asociaciones (Fig. 6–9)
│   ├── 05_eda_automatizado.ipynb          # NUEVO · skimpy + fg-data-profiling sobre muestra
│   └── legacy/                            # 05_preprocessing_pipelines, 06_clustering_kmeans, 07_clustering_dbscan,
│                                          # 08_model_comparison_and_interpretation: se MUEVEN aquí sin cambiar su lógica
│                                          # (solo imports); la U3 los reemplaza (ver §5.4)
├── scripts/
│   ├── descargar_datos.py        # NUEVO · descarga zip de UCI, verifica hash, convierte a Parquet
│   ├── figuras_u2.py             # NUEVO · genera las Figuras 6–9 de la spec en reports/figures/
│   └── perfil.py                 # NUEVO · perfil HTML sobre muestra (seed=42) en reports/
├── src/segmentacion/             # renombrar el paquete actual `src/` → `src/segmentacion/` (importable)
│   ├── config.py                 # rutas + FECHA_SNAPSHOT (U2) ; la U3 añade FECHA_CORTE, HORIZONTE, SEMILLA
│   ├── data/cargar.py            # leer_excel_con_esquema(), leer_parquet(), ESQUEMA
│   ├── data/limpieza.py          # quitar_duplicados, separar_cancelaciones, exigir_cliente, filtrar_validas, preparar_ventas
│   ├── features/rfm.py           # PORTAR a Polars con firma construir_rfm(ventas, fecha_corte) (§5.3); la U3 la reutiliza
│   ├── features/preprocessing.py # SIN CAMBIOS de lógica (pandas): lo usan los cuadernos legacy y la app
│   ├── models/{clustering,profiling}.py, utils/logger.py   # SIN CAMBIOS de lógica; solo se mueven con el paquete
│   ├── eda/resumen.py            # cuartiles(), cercas_tukey(), asociaciones(), ventas_por_mes(), matriz_dia_hora()
│   └── visualization/plots.py    # PORTAR (matplotlib recibe Polars → .to_numpy())
├── tests/
│   ├── test_limpieza.py          # NUEVO · 5 filas inventadas por regla
│   └── test_cifras_referencia.py # NUEVO · marca @pytest.mark.slow; verifica §4 sobre el Parquet real si existe
├── reports/                      # gitignored salvo .gitkeep
└── pyproject (raíz)              # extra `segmentacion-clientes` += polars>=2.0, fastexcel, skimpy, fg-data-profiling
                                  # (ver §5.5: pin de scikit-learn y configuración de pytest)
```

### 5.2 Decisiones de diseño

| Decisión | Alternativas | Elección y razón |
|---|---|---|
| Lectura del Excel | `ucimlrepo` (devuelve pandas); `pl.read_excel` con calamine/fastexcel | `pl.read_excel(sheet_id=0, schema_overrides={"Invoice": pl.Utf8, "StockCode": pl.Utf8})` — es la lección pedagógica del curso («declarar el esquema») |
| Formato intermedio | CSV; Parquet | Parquet en `data/raw/online_retail_II.parquet`; permite enseñar `scan_parquet` |
| Dónde viven las reglas | En el cuaderno; en `src/` | En `src/segmentacion/data/limpieza.py`, una función por regla, `preparar_ventas` las encadena con `.pipe` y registra `n_antes/n_después` con `logging` |
| Cancelaciones | Eliminar; conservar | Conservar en DataFrame aparte y escribir `data/processed/cancelaciones.parquet` (tasa de devolución = variable candidata para U3) |
| Modo eager/lazy | Todo eager; todo lazy | Cuadernos eager (ver resultados); funciones de `src/` aceptan `pl.DataFrame` y `pl.LazyFrame` (tipado `pl.DataFrame | pl.LazyFrame`) y `scripts/` usan lazy |
| Perfil automático | ydata-profiling; Sweetviz | `fg-data-profiling` (nombre actual del paquete) con `minimal=True` sobre `sample(n=100_000, seed=42).to_pandas()`; `skim` directo sobre Polars |
| Gráficos | Plotly; Matplotlib | Matplotlib con paleta `#4F69CC/#ED7D31/#2E7D32`, mismos títulos que las figuras de la spec |

### 5.3 Contratos de las funciones clave

```python
# src/segmentacion/data/cargar.py
ESQUEMA = {"Invoice": pl.Utf8, "StockCode": pl.Utf8, "Customer ID": pl.Int64}
def leer_excel_con_esquema(ruta: Path) -> pl.DataFrame: ...   # concatena hojas, aplica ESQUEMA
def leer_parquet(ruta: Path, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame: ...

# src/segmentacion/data/limpieza.py
def quitar_duplicados(df): ...
def separar_cancelaciones(df) -> tuple[df, df]: ...   # (sin_cancelaciones, cancelaciones)
def exigir_cliente(df): ...
def filtrar_validas(df): ...                           # Quantity > 0 & Price > 0
def preparar_ventas(df, log: bool = True) -> tuple[df, df]: ...   # añade Importe; devuelve (ventas, cancelaciones)

# src/segmentacion/features/rfm.py  — firma pensada para que la U3 la reutilice sin cambios
def construir_rfm(ventas, fecha_corte: datetime) -> pl.DataFrame: ...
#   usa solo ventas con InvoiceDate < fecha_corte; columnas: Customer ID, recencia, antiguedad,
#   frecuencia, monetario, productos_distintos. En la U2 se llama con FECHA_SNAPSHOT = máx(InvoiceDate) + 1 día.

# src/segmentacion/eda/resumen.py
def cuartiles(df, col) -> dict
def cercas_tukey(df, col, k=1.5) -> tuple[float, float, float]   # (inferior, superior, fraccion_fuera)
def asociaciones(df, a, b) -> dict                               # {"pearson":…, "spearman":…}
def ventas_por_mes(df) -> pl.DataFrame
def matriz_dia_hora(df) -> pl.DataFrame
```

### 5.4 Numeración de cuadernos: un solo mapa para U2 y U3

| Cuaderno actual | Tras esta spec (U2) | Tras `SPEC_procesamiento_U3.md` |
|---|---|---|
| — | `00_objetivo_de_analisis` (nuevo) | igual |
| `01_data_understanding` | `01_comprension_datos` (portado a Polars) | igual |
| `02_data_cleaning` | `02_limpieza` (portado) | igual |
| `03_feature_engineering` | `03_variables_cliente` (portado; usa `construir_rfm`) | igual |
| `04_exploratory_data_analysis` | `04_eda` (portado) | igual |
| — | `05_eda_automatizado` (nuevo) | igual |
| `05_preprocessing_pipelines` | `legacy/` (solo imports) | se reemplaza por `07_ingenieria_caracteristicas`; se elimina de `legacy/` |
| `06_clustering_kmeans`, `07_clustering_dbscan`, `08_model_comparison_and_interpretation` | `legacy/` (solo imports) | se fusionan en `10_segmentacion_no_supervisada`; se eliminan de `legacy/` |
| — | — | `06_particion_y_fuga`, `07_ingenieria_caracteristicas`, `08_llm_como_transformador`, `09_modelos_y_pipelines` (nuevos) |

`app/streamlit_app.py` sigue funcionando en ambos pasos (solo cambian los `import`). Si se conserva `build_notebooks.py`, debe generar exactamente estos nombres.

### 5.5 Configuración del repositorio que esta spec debe dejar lista

- `pyproject.toml` raíz: la dependencia base está fijada en `scikit-learn>=1.3.0,<1.6.0`. Subirla a `scikit-learn>=1.9` (las guías citan 1.9.1 y la U3 verifica la versión al cargar modelos); comprobar que `titanic`, `meli` y `regresion-automoviles` siguen importando.
- `[tool.pytest.ini_options]` raíz apunta a `testpaths = ["tests"]` (no existe) y a `--cov=src`. Cambiar a `testpaths = ["projects/00-segmentacion-clientes/tests"]`, registrar el marcador `slow` y ajustar `--cov` al paquete `segmentacion` (o quitar la cobertura de `addopts`).
- Python del proyecto: `.python-version` = 3.11. Las cifras se calcularon con Python 3.13; si alguna difiere en 3.11, reportarlo (no debería: no dependen del intérprete).
- `scripts/figuras_u2.py` llega con copias propias de las funciones de limpieza: tras el paso 2 debe importarlas de `segmentacion.data.limpieza` (una sola fuente de verdad).

## 6. Plan de implementación (orden sugerido)

0. Renombrar `src/` → `src/segmentacion/`, mover los cuadernos 05–08 a `notebooks/legacy/` y actualizar `import` en ellos, en `app/` y en `build_notebooks.py`; ajustar `pyproject.toml` según §5.5. Verificar que la app de Streamlit arranca.
1. `scripts/descargar_datos.py` + `data/README.md` (procedencia, licencia, hash). Verificar que `uv run python scripts/descargar_datos.py` deja el Parquet.
2. `src/segmentacion/data/{cargar,limpieza}.py` + `tests/test_limpieza.py`. `uv run pytest` en verde.
3. `tests/test_cifras_referencia.py` contra §4 (marcado `slow`).
4. Portar cuadernos 01 → 02 → 04 → 03 (en ese orden: el EDA depende de la limpieza, RFM es puente a U3). Regenerar con `build_notebooks.py` si se mantiene ese flujo; si no, eliminarlo del proyecto para no tener dos fuentes de verdad.
5. Nuevos cuadernos 00 y 05.
6. `scripts/figuras_u2.py` y `scripts/perfil.py`; comprobar que las figuras coinciden con las de la spec.
7. `HALLAZGOS.md`, README (sección de objetivo), `GUIA_LECTURA.md`, `DICCIONARIO_DATOS.md`.
8. `pyproject.toml` raíz: extra `segmentacion-clientes`; añadir `nbstripout` a pre-commit; quitar `black` (ruff format ya cubre).
9. Ensayo completo del guion del videotutorial con cronómetro (≤ 15 min).

## 7. Criterios de aceptación

- [ ] `uv sync --extra segmentacion-clientes && uv run python scripts/descargar_datos.py` deja `data/raw/online_retail_II.parquet` con 1 067 371 filas y `Invoice` de tipo `String`.
- [ ] `uv run pytest tests/test_limpieza.py` pasa en < 5 s sin datos reales.
- [ ] `uv run pytest -m slow` reproduce todas las cifras de §4 (tolerancia 0 en conteos; 0,001 en correlaciones).
- [ ] Los cuadernos 00–05 se ejecutan en orden con `uv run jupyter nbconvert --execute` sin errores y sin `!pip`/`!wget`.
- [ ] Ningún cuaderno importa `pandas` salvo 05 (perfil HTML) y, dentro de él, solo vía `.to_pandas()` sobre una muestra.
- [ ] `scripts/figuras_u2.py` produce `reports/figures/f4_ventas.png`, `f5_precio.png`, `f6_qty_price.png`, `f9_heatmap.png` con los mismos títulos de la spec.
- [ ] `HALLAZGOS.md` contiene las siete filas de la Tabla 7 de la spec.
- [ ] `GUIA_LECTURA.md` separa «Unidad 2 (00–05)» de «Unidad 3 (06–10, pendiente)» con tiempos estimados; los cuadernos de `legacy/` se marcan como «en migración».
- [ ] Los cuadernos de `legacy/` y `app/streamlit_app.py` siguen ejecutándose tras el cambio de paquete (`uv run streamlit run app/streamlit_app.py` arranca sin errores de importación).
- [ ] `construir_rfm(ventas, fecha_corte)` existe con esa firma y solo usa ventas anteriores al corte (prueba con 5 filas, una posterior al corte).
- [ ] `scikit-learn>=1.9` en el `pyproject.toml` y `pytest` configurado según §5.5.
- [ ] README raíz del repositorio lista los cinco proyectos (hoy solo lista Titanic).
- [ ] Ruff y pre-commit en verde; `.gitignore` excluye `data/raw/*.xlsx`, `*.parquet`, `reports/*.html`.

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| UCI republica el archivo y cambian las cifras | Hash en `data/README.md`; prueba `slow` falla y avisa; actualizar material del curso y spec a la vez |
| Lectura del Excel lenta en clase (~4–5 s con fastexcel; mucho más con openpyxl) | Parquet como formato de trabajo; el Excel se lee una vez en `descargar_datos.py` |
| `fg-data-profiling` tarda minutos sobre 1 M de filas | Muestra de 100 000 filas con semilla; `minimal=True` |
| Dos fuentes de verdad (`build_notebooks.py` vs. `.ipynb`) | Decidir una; si se conserva el generador, los `.ipynb` no se editan a mano |
| El guion del video se grabó antes del port | El guion está escrito sobre decisiones, no sobre nombres de celdas; cambian los nombres de funciones, no el relato |

## 9. Preguntas abiertas

1. ¿El cuaderno 03 (RFM) se queda en la Unidad 2 como «construcción de variables para explorar» o se mueve a la Unidad 3? Propuesta: se queda, porque la Tabla 7 de la spec ya lo anuncia como puente; la función `construir_rfm` con fecha de corte es la misma que la U3 reutiliza, así que no hay duplicación.
2. ¿Se publica el repositorio con los datos crudos incluidos (como hace `prediccion-precios-automoviles`) o solo con el script de descarga? Propuesta: solo script + hash, coherente con lo que enseña la Unidad 1.
3. ¿Nombre del paquete: `segmentacion` o `segmentacion_clientes`? Propuesta: `segmentacion`.

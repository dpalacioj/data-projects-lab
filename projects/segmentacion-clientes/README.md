# Segmentación de Clientes con Aprendizaje No Supervisado

Proyecto educativo que implementa un flujo completo de **segmentación de clientes**
usando **K-Means** y **DBSCAN** sobre un dataset real de e-commerce. Está pensado
como tutorial guiado para estudiantes de maestría en IA: cada notebook explica el
*qué*, el *por qué* y los errores comunes.

---

## Tabla de Contenido

- [Documentación Rápida](#documentación-rápida)
- [Caso de Negocio](#caso-de-negocio)
- [Objetivo de negocio, objetivo de análisis y criterio de éxito](#objetivo-de-negocio-objetivo-de-análisis-y-criterio-de-éxito)
- [Objetivo Pedagógico](#objetivo-pedagógico)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Flujo del Proyecto](#flujo-del-proyecto)
- [Conceptos Aprendidos](#conceptos-aprendidos)
- [Posibles Extensiones](#posibles-extensiones)
- [Recomendaciones para Estudiantes](#recomendaciones-para-estudiantes)
- [Licencia y Autoría](#licencia-y-autoría)

---

## Documentación Rápida

- **[GUIA_LECTURA.md](GUIA_LECTURA.md)** — orden recomendado de lectura.
- **[DICCIONARIO_DATOS.md](DICCIONARIO_DATOS.md)** — variables del dataset y features derivadas.

---

## Caso de Negocio

Una empresa de retail online (B2C) con miles de clientes en distintos países
quiere **personalizar sus campañas de marketing**. La pregunta de negocio es:

> *¿Cómo agrupamos a nuestros clientes en perfiles homogéneos para diseñar
> ofertas, campañas de fidelización y planes de reactivación específicos?*

Sin segmentación, todos los clientes reciben el mismo correo promocional.
Eso es ineficiente: un cliente VIP no necesita el mismo descuento que uno
inactivo, y un cliente nuevo no debería recibir un email de "te extrañamos".

Preguntas concretas que el proyecto busca responder:

1. ¿Qué tipos de clientes existen? ¿Cuántos hay de cada tipo?
2. ¿Qué clientes generan más valor (Champions)?
3. ¿Qué clientes están en riesgo de irse?
4. ¿Qué clientes responderían mejor a una promoción agresiva?
5. ¿Hay clientes con comportamiento atípico (mayoristas, fraudes)?

---

## Objetivo de negocio, objetivo de análisis y criterio de éxito

Reproduce la **Tabla 2** del OVA de la Unidad 2 — la misma matriz que
usamos para alinear negocio y análisis antes de tocar código.

| Campo | Contenido |
|---|---|
| **Problema de negocio** | El retailer trata a todos los clientes por igual; las campañas de regalo rinden poco porque no hablan al cliente adecuado en el momento adecuado. |
| **Pregunta de negocio** | ¿Cómo agrupar a los clientes para que las acciones de marketing (comunicación, descuentos, retención) tengan rendimiento diferenciado? |
| **Objetivo de análisis** | Construir segmentos no supervisados de clientes basados en RFM extendido, estables entre K-Means y DBSCAN, acompañados de una ficha accionable por segmento. |
| **Criterio de éxito analítico** | Silhouette ≥ 0,35 y más del 70 % de los clientes concentrados en 3-5 segmentos interpretables. |
| **Criterio de éxito de negocio** | Que marketing adopte las fichas de acción (medible en U4 por tasa de contacto por segmento). |
| **Riesgo principal** | Que el pipeline no sea reproducible el día de grabación. Mitigación: `scripts/figuras_u2.py` y `tests/test_cifras_referencia.py` fallan si cambia alguna cifra del OVA. |
| **Datos y licencia** | Online Retail II (UCI 502, Chen 2019, CC BY 4.0). Procedencia, hash y procedimiento en [`data/README.md`](data/README.md). |

El cuaderno `00_objetivo_de_analisis.ipynb` deja esta misma tabla al
alcance en vivo durante clase.

---

## Objetivo Pedagógico

Al terminar el proyecto, un estudiante debe ser capaz de:

- Formular un problema de segmentación a partir de un caso de negocio.
- Limpiar un dataset transaccional real con problemas típicos (faltantes,
  cancelaciones, duplicados, outliers).
- Construir variables RFM (*Recency*, *Frequency*, *Monetary*) y extenderlas
  con métricas de comportamiento.
- Diseñar pipelines de preprocesamiento con `Pipeline` y `ColumnTransformer`.
- Aplicar **K-Means** y **DBSCAN**, justificando hiperparámetros.
- Evaluar clustering con **Silhouette**, **Davies-Bouldin**,
  **Calinski-Harabasz** e **Inertia**.
- Interpretar segmentos en términos de negocio y proponer acciones.
- Construir una **demo interactiva en Streamlit**.
- Organizar el código siguiendo buenas prácticas (modular, documentado, reproducible).

---

## Dataset

**Online Retail II** (UCI Machine Learning Repository, ID 502).

- **Fuente:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Cobertura temporal:** 2009-12-01 a 2011-12-09.
- **Tamaño:** ~1 millón de transacciones, ~5,800 clientes únicos.
- **Origen:** retailer británico de regalos *all-occasion*, principalmente B2C
  pero con clientes mayoristas.
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### ¿Por qué este dataset sirve para segmentación?

1. **Es transaccional**: cada fila es una línea de factura, lo que permite
   agrupar por cliente y derivar métricas RFM auténticas.
2. **Tiene problemas reales**: faltantes en `CustomerID`, cancelaciones
   (facturas con prefijo `C`), precios negativos, outliers de cantidades.
   Es perfecto para enseñar limpieza con un caso real, no de juguete.
3. **Es lo bastante grande** (~1M filas) para que las técnicas de clustering
   muestren patrones interesantes, pero no tan grande como para necesitar
   GPU o Spark.
4. **Tiene contexto claro**: cada cliente = un caso de uso de negocio
   con ofertas, retención y campañas potenciales.

Ver [DICCIONARIO_DATOS.md](DICCIONARIO_DATOS.md) para el detalle de cada variable.

---

## Estructura del Proyecto

```
segmentacion-clientes/
│
├── data/
│   ├── README.md      # Procedencia, licencia (CC BY 4.0) y SHA-256
│   ├── raw/           # Descargado por scripts/descargar_datos.py (gitignored)
│   ├── interim/       # Datos limpios (intermedios)
│   └── processed/     # ventas_validas, cancelaciones, rfm_u2 (gitignored)
│
├── notebooks/
│   ├── 00_objetivo_de_analisis.ipynb      # CRISP-DM fase 1 (nuevo)
│   ├── 01_comprension_datos.ipynb         # Lectura con esquema + diagnóstico
│   ├── 02_limpieza.ipynb                  # preparar_ventas → cancelaciones.parquet
│   ├── 03_variables_cliente.ipynb         # construir_rfm con fecha_corte
│   ├── 04_eda.ipynb                       # Cuartiles, Tukey, correlaciones
│   ├── 05_eda_automatizado.ipynb          # skimpy + fg-data-profiling
│   └── legacy/
│       ├── 05_preprocessing_pipelines.ipynb      # ↓ en migración a U3
│       ├── 06_clustering_kmeans.ipynb
│       ├── 07_clustering_dbscan.ipynb
│       └── 08_model_comparison_and_interpretation.ipynb
│
├── scripts/
│   ├── descargar_datos.py     # Descarga + SHA-256 + xlsx → parquet
│   ├── figuras_u2.py          # Figuras 6-9 del OVA en reports/figures/
│   ├── perfil.py              # Perfil HTML sobre muestra (seed=42)
│   ├── caso_recompra.py       # Insumo de U3 (recompra 90 d)
│   └── figuras_u3.py          # Insumo de U3 (ROC, PR, importancias)
│
├── src/segmentacion/          # Paquete importable
│   ├── config.py                 # Rutas, FECHA_SNAPSHOT
│   ├── data/
│   │   ├── cargar.py             # ESQUEMA, leer_excel_con_esquema, leer_parquet
│   │   ├── limpieza.py           # quitar_duplicados, separar_cancelaciones,
│   │   │                         # exigir_cliente, filtrar_validas, preparar_ventas
│   │   └── loader.py             # (pandas, lo usa el legacy)
│   ├── eda/
│   │   └── resumen.py            # cuartiles, cercas_tukey, asociaciones, etc.
│   ├── features/
│   │   ├── rfm.py                # construir_rfm(ventas, fecha_corte)
│   │   └── preprocessing.py      # Pipeline sklearn (legacy / U3)
│   ├── models/{clustering,profiling}.py   # Sin cambios (legacy / U3)
│   ├── utils/logger.py
│   └── visualization/plots.py
│
├── tests/
│   ├── test_limpieza.py          # Rápido, 6 filas inventadas
│   ├── test_rfm.py               # Rápido, corte explícito
│   └── test_cifras_referencia.py # @pytest.mark.slow, cifras del §4 OVA
│
├── app/streamlit_app.py          # Demo interactiva (consume U3 output)
├── models/                        # Modelos entrenados (.joblib, gitignored)
├── reports/figures/               # Gráficos generados (gitignored)
├── HALLAZGOS.md                   # Hallazgos U2 → insumo de U3
├── GUIA_LECTURA.md
├── DICCIONARIO_DATOS.md
└── .gitignore
```

> **Nota:** `src/segmentacion/models/` contiene **código** para
> clustering; `models/` (en la raíz) contiene los **artefactos**
> entrenados. Es la convención estándar en proyectos de ML.

---

## Instalación

### Requisitos

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) como gestor de paquetes (recomendado).

### Paso 1 · Instalar dependencias

Desde la raíz del repositorio `data-projects-lab`:

```bash
uv sync --extra segmentacion-clientes
```

Esto instala `polars>=2.0`, `fastexcel`, `skimpy`, `fg-data-profiling`,
`streamlit`, `plotly`, `openpyxl` (lectura de .xlsx), `joblib` y
`pyarrow`, además de las dependencias base
(pandas, numpy, scikit-learn>=1.9, matplotlib, seaborn, jupyter).

Para los tests y herramientas de desarrollo:

```bash
uv sync --extra segmentacion-clientes --extra dev
```

### Paso 2 · Activar el entorno virtual

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Paso 3 · Descargar el dataset

```bash
uv run python scripts/descargar_datos.py
```

> El script descarga el `.zip` desde UCI, verifica el **SHA-256**
> contra la constante del propio script, extrae `online_retail_II.xlsx`
> y escribe `data/raw/online_retail_II.parquet` con el esquema del OVA.
> Si el hash no coincide, se detiene (código 2). Detalles en
> [`data/README.md`](data/README.md).

---

## Cómo Ejecutar

### Notebooks (Unidad 2)

```bash
cd projects/segmentacion-clientes
uv run jupyter lab
```

Abrir los notebooks en orden: `00_objetivo_de_analisis` →
`01_comprension_datos` → `02_limpieza` → `04_eda` →
`03_variables_cliente` → `05_eda_automatizado`.

### Notebooks (Unidad 3)

El caso supervisado "recompra en 90 días" + ingeniería de pipelines sin
fuga. Orden recomendado: **06 → 07 → (08) → 09 → 10**.

- `06_particion_y_fuga`: cohortes A y B con `cohorte(ventas, cancelaciones, corte, horizonte=90)`; demostración de fuga temporal.
- `07_ingenieria_caracteristicas`: anatomía del `ColumnTransformer`.
- `08_llm_como_transformador` (opcional): categorización con LLM; `FakeLLM` por defecto. Ver `SPEC_llm_transformador.md`.
- `09_modelos_y_pipelines`: tres modelos + regla; umbrales; importancia; curva de profundidad; `guardar()` el bosque en `models/recompra_v1/`.
- `10_segmentacion_no_supervisada`: k-means + DBSCAN sobre el mismo preprocesador.

Scripts asociados:

```bash
uv run python scripts/caso_recompra.py       # genera reports/{resultados.json, cohorte_A.parquet, cohorte_B.parquet}
uv run python scripts/figuras_u3.py          # genera reports/figures/g_{fuga,sobreajuste,roc_pr,importancia}.png
```

### Pruebas

```bash
uv run pytest                                # rápido (sin red, sin claves de API; incluye FakeLLM del módulo LLM)
uv run pytest -m slow                        # cifras del OVA contra el parquet real (U2 §4 + U3 §9)
```

### Scripts auxiliares (U2)

```bash
uv run python scripts/figuras_u2.py data/raw/online_retail_II.parquet reports/figures
uv run python scripts/perfil.py              # HTML en reports/perfil_muestra.html
```

### Aplicación Streamlit

```bash
cd projects/segmentacion-clientes
uv run streamlit run app/streamlit_app.py
```

La app se abre en http://localhost:8501. En U4 consumirá
`models/recompra_v1/metadatos.json` (contrato con la ficha del modelo).

---

## Flujo del Proyecto

```
   Datos Crudos                           Datos Limpios
   (Excel UCI)                            (parquet)
        |                                      |
        | 01_data_understanding                | 02_data_cleaning
        v                                      v
   ┌────────────┐    03_feature_engineering   ┌──────────────┐
   │ ~1M filas  │ ───────────────────────────>│ Features RFM │
   │ ~5.8k cust.│                             │ por cliente  │
   └────────────┘                             └──────────────┘
                                                    |
        04_EDA  +  05_pipelines  ───────────────────|
                                                    v
                                            ┌────────────────┐
                          06_kmeans  ──────>│  Clusters      │
                          07_dbscan  ──────>│  K-Means+DBSCAN│
                                            └────────────────┘
                                                    |
                          08_interpretation  ───────|
                                                    v
                                            ┌────────────────┐
                                            │ Perfiles +     │
                                            │ Acciones       │
                                            │ + Streamlit    │
                                            └────────────────┘
```

---

## Conceptos Aprendidos

| Etapa | Conceptos |
|---|---|
| **Comprensión** | EDA inicial, calidad de datos, tipos de variables. |
| **Limpieza** | Faltantes, duplicados, outliers, decisiones justificadas. |
| **Features** | RFM, agregaciones, métricas de comportamiento. |
| **Pipelines** | `Pipeline`, `ColumnTransformer`, `FunctionTransformer`. |
| **Escalado** | Por qué es crítico para K-Means y DBSCAN. |
| **K-Means** | Centroides, inertia, codo, silueta. |
| **DBSCAN** | `eps`, `min_samples`, ruido, formas no esféricas. |
| **Métricas** | Silhouette, Davies-Bouldin, Calinski-Harabasz. |
| **PCA** | Reducción de dimensionalidad para visualización. |
| **Interpretación** | De clusters numéricos a perfiles de negocio. |
| **Demo** | Streamlit para presentar resultados a stakeholders. |

---

## Posibles Extensiones

- Agregar **Agglomerative Clustering** y **GMM** para comparar.
- Probar **t-SNE** o **UMAP** para visualización 2D no lineal.
- Implementar **HDBSCAN** (versión jerárquica de DBSCAN).
- Calcular **Customer Lifetime Value (CLV)** y agregarlo como feature.
- Crear un **dashboard MLflow** para experimentos.
- Hacer la app de Streamlit **multi-página** con secciones separadas.
- Agregar tests con `pytest` para los módulos en `src/`.

---

## Recomendaciones para Estudiantes

1. **No saltes los notebooks.** Cada uno construye sobre el anterior.
2. **Lee primero el Markdown.** El código tiene sentido cuando entiendes el por qué.
3. **Cambia los hiperparámetros y observa.** El aprendizaje viene de experimentar.
4. **Dialoga con el código.** ¿Qué pasa si quito el escalado? ¿Si uso `eps=2.0`?
5. **No confíes en una sola métrica.** El silhouette máximo no siempre da el
   "mejor" k. Combínalo con interpretación de negocio.
6. **Documenta tus decisiones.** En un proyecto real, justificar es tan
   importante como ejecutar.

---

## Licencia y Autoría

- **Código:** MIT License (mismo del repositorio raíz).
- **Dataset:** CC BY 4.0 (Online Retail II, D. Chen, UCI ML Repository).
- **Autor:** David Palacio Jiménez · davidpalacioj@gmail.com

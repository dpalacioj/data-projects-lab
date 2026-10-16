# Segmentación de Clientes con Aprendizaje No Supervisado

Proyecto educativo que implementa un flujo completo de **segmentación de clientes**
usando **K-Means** y **DBSCAN** sobre un dataset real de e-commerce. Está pensado
como tutorial guiado para estudiantes de maestría en IA: cada notebook explica el
*qué*, el *por qué* y los errores comunes.

---

## Tabla de Contenido

- [Documentación Rápida](#documentación-rápida)
- [Caso de Negocio](#caso-de-negocio)
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
│   ├── raw/           # Dataset original (descargado por src/data/loader.py)
│   ├── interim/       # Datos limpios (intermedios)
│   └── processed/     # Features finales y clientes segmentados
│
├── notebooks/         # Tutoriales paso a paso (ejecutar en orden)
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_preprocessing_pipelines.ipynb
│   ├── 06_clustering_kmeans.ipynb
│   ├── 07_clustering_dbscan.ipynb
│   └── 08_model_comparison_and_interpretation.ipynb
│
├── src/               # Código modular reutilizable
│   ├── config.py             # Constantes y rutas
│   ├── data/
│   │   └── loader.py         # Descarga y carga del dataset
│   ├── features/
│   │   ├── rfm.py            # build_rfm, build_customer_features
│   │   └── preprocessing.py  # Pipelines sklearn
│   ├── models/
│   │   ├── clustering.py     # fit_kmeans, fit_dbscan, métricas
│   │   └── profiling.py      # Perfilamiento de segmentos
│   ├── visualization/
│   │   └── plots.py          # Gráficos reutilizables
│   └── utils/
│       └── logger.py         # Logger estandarizado
│
├── app/
│   └── streamlit_app.py      # Demo interactiva
│
├── models/             # Modelos entrenados (.joblib)
├── reports/figures/    # Gráficos generados
│
├── README.md
├── GUIA_LECTURA.md
├── DICCIONARIO_DATOS.md
└── .gitignore
```

> **Nota:** `src/models/` contiene **código** para clustering;
> `models/` (en la raíz) contiene los **artefactos** entrenados.
> Es la convención estándar en proyectos de ML.

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

Esto instala `streamlit`, `plotly`, `openpyxl` (lectura de .xlsx),
`joblib` y `pyarrow`, además de las dependencias base
(pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter).

### Paso 2 · Activar el entorno virtual

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Paso 3 · Descargar el dataset

```bash
uv run python -c "from segmentacion.data.loader import download_dataset; download_dataset()"
```

> El archivo (~45 MB descomprimido) queda en `data/raw/online_retail_II.xlsx`.
> Si ya existe, no se vuelve a descargar.

---

## Cómo Ejecutar

### Notebooks

```bash
cd projects/segmentacion-clientes
jupyter lab
```

Abrir los notebooks en orden numérico (`01_` → `08_`).
Cada uno guarda salidas que el siguiente consume.

### Aplicación Streamlit

Después de ejecutar al menos hasta el notebook `06_clustering_kmeans.ipynb`:

```bash
cd projects/segmentacion-clientes
uv run streamlit run app/streamlit_app.py
```

La app se abre en http://localhost:8501.

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

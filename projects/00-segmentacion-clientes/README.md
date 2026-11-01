---
title: "00 · Segmentación de Clientes (caso guía)"
parent: Proyectos
nav_order: 0
---
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

Matriz que usamos para alinear negocio y análisis antes de tocar código.

| Campo | Contenido |
|---|---|
| **Problema de negocio** | El retailer trata a todos los clientes por igual; las campañas de regalo rinden poco porque no hablan al cliente adecuado en el momento adecuado. |
| **Pregunta de negocio** | ¿Cómo agrupar a los clientes para que las acciones de marketing (comunicación, descuentos, retención) tengan rendimiento diferenciado? |
| **Objetivo de análisis** | Construir segmentos no supervisados de clientes basados en RFM extendido, estables entre K-Means y DBSCAN, acompañados de una ficha accionable por segmento. |
| **Criterio de éxito analítico** | Silhouette ≥ 0,35 y más del 70 % de los clientes concentrados en 3-5 segmentos interpretables. |
| **Criterio de éxito de negocio** | Que marketing adopte las fichas de acción (medible en U4 por tasa de contacto por segmento). |
| **Riesgo principal** | Que el pipeline no sea reproducible el día de grabación. Mitigación: `scripts/figuras_u2.py` y `tests/test_cifras_referencia.py` fallan si cambia alguna cifra de referencia. |
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
│   ├── 06_particion_y_fuga.ipynb          # Cohortes A/B, fuga temporal
│   ├── 07_ingenieria_caracteristicas.ipynb # ColumnTransformer de la Tabla 4
│   ├── 08_llm_como_transformador.ipynb    # Placeholder (modulo retirado)
│   ├── 09_modelos_y_pipelines.ipynb       # Regresion/Arbol/Bosque + umbral
│   └── 10_segmentacion_no_supervisada.ipynb # k-means + DBSCAN + perfiles
│
├── scripts/
│   ├── descargar_datos.py     # Descarga + SHA-256 + xlsx → parquet
│   ├── figuras_u2.py          # Figuras 6-9 de referencia en reports/figures/
│   ├── perfil.py              # Perfil HTML sobre muestra (seed=42)
│   ├── caso_recompra.py       # Insumo de U3 (recompra 90 d)
│   ├── figuras_u3.py          # Insumo de U3 (ROC, PR, importancias)
│   └── entrenar_recompra.py   # U4: entrena y guarda models/recompra_v1/
│
├── src/segmentacion/          # Paquete importable
│   ├── config.py                 # Rutas, FECHA_SNAPSHOT, RUTA_MODELO (U4)
│   ├── data/
│   │   ├── cargar.py             # ESQUEMA, leer_excel_con_esquema, leer_parquet
│   │   ├── limpieza.py           # quitar_duplicados, separar_cancelaciones,
│   │   │                         # exigir_cliente, filtrar_validas, preparar_ventas
│   │   └── loader.py             # (pandas, lo usa la app Streamlit)
│   ├── eda/
│   │   └── resumen.py            # cuartiles, cercas_tukey, asociaciones, etc.
│   ├── features/
│   │   ├── rfm.py                # construir_rfm(ventas, fecha_corte)
│   │   └── preprocessing.py      # Pipeline sklearn (la usa la app Streamlit)
│   ├── models/{clustering,profiling}.py   # K-Means/DBSCAN + perfiles (cuaderno 10 + app)
│   ├── servicio/                 # U4: Predictor + EntradaCliente (nucleo de prediccion)
│   ├── utils/logger.py
│   └── visualization/plots.py
│
├── tests/
│   ├── test_limpieza.py          # Rápido, 6 filas inventadas
│   ├── test_rfm.py               # Rápido, corte explícito
│   ├── test_cifras_referencia.py # @pytest.mark.slow, cifras de referencia U2
│   └── servicio/                 # U4: tests del nucleo y de las interfaces
│
├── app/
│   ├── streamlit_app.py          # Demo interactiva (consume U3 output)
│   └── recompra_streamlit.py     # U4: interfaz Streamlit sobre models/recompra_v1/
├── api/main.py                    # U4: API FastAPI sobre el mismo Predictor
├── examples/                      # U4: CSV y .http de ejemplo
├── models/                        # Modelos entrenados (.joblib, gitignored)
│   └── recompra_v1/MODEL_CARD.md  # U4: ficha del modelo (uso previsto, limitaciones)
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
# o, desde la raíz del repo:
make descargar-datos
```

> El script descarga el `.zip` desde UCI, verifica el **SHA-256**
> contra la constante del propio script, extrae `online_retail_II.xlsx`
> y escribe `data/raw/online_retail_II.parquet` con el esquema declarado.
> Si el hash no coincide, se detiene (código 2). Detalles en
> [`data/README.md`](data/README.md).

### Paso 4 · Preparar los parquets de trabajo

```bash
# desde la raíz del repo
make preparar-datos            # ventas_validas.parquet + cancelaciones.parquet
make features-segmentacion     # features_clientes.parquet (app no supervisada)
make train-recompra            # models/recompra_v1/ (artefacto U3 + U4)
```

Equivalentes manuales:

```bash
cd projects/00-segmentacion-clientes
uv run python scripts/preparar_datos.py
uv run python scripts/generar_features_clientes.py
uv run python scripts/entrenar_recompra.py
```

Tras estos tres comandos tienes todo lo que las apps, la API y los tests
necesitan. Los notebooks se pueden abrir en cualquier orden sin regenerar
estos artefactos.

---

## Cómo Ejecutar

### Notebooks (Unidad 2)

```bash
cd projects/00-segmentacion-clientes
uv run jupyter lab
```

Abrir los notebooks en orden: `00_objetivo_de_analisis` →
`01_comprension_datos` → `02_limpieza` → `04_eda` →
`03_variables_cliente` → `05_eda_automatizado`.

### Notebooks (Unidad 3)

El caso supervisado "recompra en 90 días" + ingeniería de pipelines sin
fuga. Orden recomendado: **06 → 07 → 09 → 10** (el cuaderno `08` quedó
como placeholder y no se corre).

- `06_particion_y_fuga`: cohortes A y B con `cohorte(ventas, cancelaciones, corte, horizonte=90)`; demostración de fuga temporal.
- `07_ingenieria_caracteristicas`: anatomía del `ColumnTransformer`.
- `08_llm_como_transformador`: **placeholder** (módulo retirado del curso).
- `09_modelos_y_pipelines`: tres modelos + regla; umbrales; importancia; curva de profundidad; `guardar()` el bosque en `models/recompra_v1/`.
- `10_segmentacion_no_supervisada`: k-means + DBSCAN sobre el mismo preprocesador.

Scripts asociados:

```bash
uv run python scripts/caso_recompra.py       # genera reports/{resultados.json, cohorte_A.parquet, cohorte_B.parquet}
uv run python scripts/figuras_u3.py          # genera reports/figures/g_{fuga,sobreajuste,roc_pr,importancia}.png
```

### Pruebas

```bash
uv run pytest                                # rápido (sin red, sin claves de API)
uv run pytest -m slow                        # cifras de referencia contra el parquet real (U2 + U3)
```

### Scripts auxiliares (U2)

```bash
uv run python scripts/figuras_u2.py data/raw/online_retail_II.parquet reports/figures
uv run python scripts/perfil.py              # HTML en reports/perfil_muestra.html
```

### Aplicación Streamlit

```bash
cd projects/00-segmentacion-clientes
uv run streamlit run app/streamlit_app.py
```

La app se abre en http://localhost:8501. En U4 consumirá
`models/recompra_v1/metadatos.json` (contrato con la ficha del modelo).

### Unidad 4: interfaces de recompra sobre `models/recompra_v1/`

La U4 **envuelve** el modelo que entrenaste en U3 (bosque aleatorio para
predecir recompra en 90 días) en dos interfaces que comparten el mismo
núcleo de predicción (`src/segmentacion/servicio/`):

- **Streamlit** (`app/recompra_streamlit.py`) — pantalla para una persona
  de negocio. Un cliente a la vez *o* un CSV con muchos.
- **FastAPI** (`api/main.py`) — endpoint HTTP para que otro sistema
  (otro equipo, un notebook de marketing, un script) consulte el modelo.

Ninguna interfaz carga `pipeline.joblib` por su cuenta: ambas importan
`Predictor` desde `segmentacion.servicio`. La ficha pública del modelo
(uso previsto, limitaciones, licencia) está en
[`models/recompra_v1/MODEL_CARD.md`](models/recompra_v1/MODEL_CARD.md).

#### Paso 0 (una sola vez) · Entrenar el artefacto

Si no tienes `models/recompra_v1/pipeline.joblib`, créalo desde la raíz
del repositorio:

```bash
make train-recompra
```

Esto corre `scripts/entrenar_recompra.py`, que entrena el bosque
aleatorio de la U3 y deja dos archivos en `models/recompra_v1/`:

- `pipeline.joblib` — el modelo serializado (ignorado por git).
- `metadatos.json` — la "ficha" del artefacto: columnas en orden,
  umbral acordado, cifras (`auc_validacion`, `auc_prueba`,
  `tabla_umbrales`), versiones y `hash` del parquet usado.

> Si falla con *"no se encontró el parquet"*, primero descarga los
> datos: `cd projects/00-segmentacion-clientes && uv run python scripts/descargar_datos.py`.

#### Paso 1 · Opción A: Streamlit (para gente de negocio)

```bash
make app-recompra          # abre http://localhost:8501
```

La app tiene dos pestañas:

1. **Un cliente.** Rellenas el formulario con las 7 variables (gasto
   total, facturas, productos distintos, días desde la última compra,
   antigüedad, tasa de devolución, país) y pulsas **Predecir**. La app
   muestra la probabilidad de recompra en 90 días y la decisión
   (*Contactar* / *No contactar*) con el **umbral** que elijas en la
   barra lateral. Mover el slider del umbral cambia también los
   indicadores de clientes señalados, precisión y recall de la cohorte
   de prueba (vienen de `tabla_umbrales` en `metadatos.json`).
2. **Lista de clientes.** Subes un CSV con las 7 columnas del modelo
   (puede traer columnas extra como `cliente` o `perfil`, se preservan
   en la salida). La app devuelve la tabla ordenada por probabilidad
   descendente y un botón para descargar sólo las filas marcadas como
   *contactar*.

**Happy path para la demo en vivo.** El proyecto incluye
[`examples/clientes_ejemplo.csv`](examples/clientes_ejemplo.csv) con 6
clientes canónicos que cubren todo el espectro. Súbelo en la pestaña
*Lista de clientes* y verás esta tabla, ya ordenada:

| cliente | perfil                     | probabilidad | recompra (umbral 0,3) |
|---------|----------------------------|-------------:|-----------------------|
| c1      | VIP activo (Alemania)      | ≈ 0,96       | **Contactar**         |
| c2      | VIP estable (UK)           | ≈ 0,75       | **Contactar**         |
| c3      | Ocasional con devoluciones | ≈ 0,35       | **Contactar**         |
| c4      | Ocasional limpio           | ≈ 0,35       | **Contactar**         |
| c5      | En riesgo con devoluciones | ≈ 0,14       | No contactar          |
| c6      | Casi perdido (Francia)     | ≈ 0,05       | No contactar          |

Son los 6 casos de la Tabla 2 del OVA U4. Mover el slider del umbral a
0,4 o 0,5 muestra en vivo cómo se contrae la lista de "contactar" sin
tocar el modelo. Es el guion recomendado para la demo.

#### Paso 1 · Opción B: FastAPI (para otro sistema)

```bash
make api-recompra          # abre http://127.0.0.1:8000/docs
```

Expone dos endpoints:

- `GET /salud` — devuelve `{"estado": "ok", "version_modelo": ..., "sklearn": ...}` si el modelo cargó bien.
- `POST /predecir` — recibe JSON con los 7 campos del cliente y responde
  `{"probabilidad": ..., "recompra": true/false, "umbral": ..., "version_modelo": ...}`.

La forma más cómoda de probarla es abrir **http://127.0.0.1:8000/docs**
(el Swagger UI que FastAPI genera solo) y usar el botón *Try it out*. O
con `curl` desde otra terminal:

```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{"monetario": 2350, "frecuencia": 9, "productos_distintos": 60,
       "recencia": 12, "antiguedad": 400, "tasa_devolucion": 0,
       "pais": "United Kingdom"}'
```

El archivo [`examples/consumir_fastapi.http`](examples/consumir_fastapi.http)
tiene llamadas listas para cualquier cliente REST (extensiones de VS
Code, PyCharm, IntelliJ).

#### ¿Y si algo falla?

- *"no se encontró metadatos.json"* o *"pipeline.joblib"* → corre
  `make train-recompra`.
- *"La versión de scikit-learn no coincide"* → pasó un `uv sync` entre
  entrenamiento y consumo. Reentrena con `make train-recompra`.
- El Streamlit no abre nada en el navegador → mira la terminal donde
  lanzaste `make app-recompra`; los errores salen ahí.
- FastAPI responde `422` → la entrada no pasó la validación de Pydantic
  (algún campo negativo o faltante). El JSON de respuesta dice qué
  campo y por qué.

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

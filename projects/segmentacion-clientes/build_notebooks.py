"""
Genera los 8 notebooks pedagógicos del proyecto.

Este script se ejecuta UNA SOLA VEZ y produce los archivos .ipynb en
``notebooks/``. Después de ejecutarlo, puedes borrarlo o conservarlo
como referencia.

Uso:
    uv run python build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def md(text: str) -> dict:
    """Crea una celda Markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    }


def code(text: str) -> dict:
    """Crea una celda de código."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS_DIR / filename
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"  + {path.name}")


# ---------------------------------------------------------------------------
# Bloques reutilizables
# ---------------------------------------------------------------------------
PATH_BOOTSTRAP = """\
# Permite importar el paquete segmentacion desde el notebook
import sys
from pathlib import Path

_cwd = Path.cwd()
if _cwd.name == "legacy":
    PROJECT_ROOT = _cwd.parent.parent
elif _cwd.name == "notebooks":
    PROJECT_ROOT = _cwd.parent
else:
    PROJECT_ROOT = _cwd
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
"""


# ===========================================================================
# 01 — Data Understanding
# ===========================================================================
def build_01():
    cells = [
        md(
            """\
# 01 · Comprensión de los Datos

> **Objetivo:** entender qué tenemos en el dataset *Online Retail II* antes
> de tocar nada. Toda decisión de limpieza y modelado depende de este paso.

## ¿Qué vamos a hacer?

1. Descargar el dataset (si no está) y cargarlo.
2. Inspeccionar dimensiones, tipos de datos y memoria.
3. Identificar variables clave para segmentación (RFM).
4. Detectar problemas iniciales: faltantes, valores extraños, cancelaciones.

## Concepto teórico breve

El **Customer Relationship Management (CRM)** moderno se basa en la idea
de que **no todos los clientes son iguales**. La segmentación busca agrupar
clientes con comportamiento similar para tratarlos de forma diferenciada.

El framework más usado en retail es **RFM**:

- **Recency**: ¿cuándo compró por última vez?
- **Frequency**: ¿con qué frecuencia compra?
- **Monetary**: ¿cuánto gasta?

Para construir RFM necesitamos un dataset **transaccional** (una fila por
operación). *Online Retail II* lo es.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd
import numpy as np

from segmentacion.data.loader import download_dataset, load_raw_transactions
from segmentacion.config import RAW_DATA_FILE

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)
"""
        ),
        md(
            """\
## 1. Descarga del dataset

La función ``download_dataset()`` descarga el ZIP desde UCI y descomprime
``online_retail_II.xlsx`` en ``data/raw/``. Si el archivo ya existe,
no vuelve a descargar.

> **Tip:** la primera ejecución toma ~30s (depende de tu red).
"""
        ),
        code("download_dataset()"),
        md(
            """\
## 2. Carga de los datos

El archivo Excel trae dos hojas (un año por hoja). El loader las concatena.

> **Tip:** ``pd.read_excel`` es lento para archivos grandes. Para uso
> productivo, después de la primera lectura conviene guardar como ``parquet``.
"""
        ),
        code(
            """\
df = load_raw_transactions()
print(f"Shape: {df.shape}")
df.head()
"""
        ),
        md("## 3. Tipos y memoria"),
        code("df.info(memory_usage='deep')"),
        md(
            """\
**Lectura del output:**

- ``Invoice`` y ``StockCode`` son objetos (string) — algunas facturas y
  productos contienen letras.
- ``CustomerID`` es ``float64`` con NaN. Tendremos que convertirlo a
  ``int`` después de eliminar los nulos.
- ``InvoiceDate`` ya viene como ``datetime64`` — no hay que parsearla.

## 4. Estadísticas descriptivas
"""
        ),
        code("df.describe(include='all').T"),
        md(
            """\
> **Errores comunes que ya saltan a la vista:**
>
> 1. ``Quantity`` y ``Price`` tienen valores **negativos**. ¿Devoluciones?
>    ¿Errores de captura? Hay que decidir qué hacer con ellos.
> 2. ``Quantity`` máximo es muchísimo mayor que la mediana → outliers
>    o clientes mayoristas.
> 3. ``CustomerID`` tiene faltantes → veremos el porcentaje exacto abajo.

## 5. Faltantes
"""
        ),
        code(
            """\
missing = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)
pd.DataFrame({"missing": missing, "pct": missing_pct}).sort_values("missing", ascending=False)
"""
        ),
        md(
            """\
**Decisión preliminar:** las filas sin ``CustomerID`` no se pueden
atribuir a un cliente, así que no sirven para RFM. Las eliminaremos
en el notebook 02. *Imputarlas sería inventar clientes*.

## 6. Cancelaciones

Las facturas que empiezan con `C` son cancelaciones (devoluciones de
mercancía). Vamos a contarlas.
"""
        ),
        code(
            """\
df["Invoice"] = df["Invoice"].astype(str)
n_cancel = df["Invoice"].str.startswith("C").sum()
print(f"Cancelaciones: {n_cancel:,} ({100*n_cancel/len(df):.2f}%)")
"""
        ),
        md(
            """\
## 7. Distribución por país
"""
        ),
        code(
            """\
df["Country"].value_counts().head(10)
"""
        ),
        md(
            """\
> **Observación:** ~90% de las transacciones son del Reino Unido. Si
> quisiéramos segmentar geográficamente, tendríamos que agrupar el resto
> en categorías ("Europa Occidental", "Otro") para que no dominen los
> clusters por sí solos.

## 8. Rango temporal
"""
        ),
        code(
            """\
print("Rango de fechas:")
print(f"  Mínimo: {df['InvoiceDate'].min()}")
print(f"  Máximo: {df['InvoiceDate'].max()}")
print(f"  Span:   {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} días")
"""
        ),
        md(
            """\
## 9. ¿Cuántos clientes tenemos?
"""
        ),
        code(
            """\
print(f"Clientes únicos (incluyendo NaN): {df['CustomerID'].nunique(dropna=False):,}")
print(f"Clientes únicos (sin NaN):       {df['CustomerID'].nunique():,}")
print(f"Productos únicos (StockCode):    {df['StockCode'].nunique():,}")
print(f"Facturas únicas:                 {df['Invoice'].nunique():,}")
"""
        ),
        md(
            """\
## Resumen del notebook

| Hallazgo | Implicación |
|---|---|
| ~25% de filas sin ``CustomerID`` | Las eliminaremos. |
| Cancelaciones (~2%) | Las eliminaremos. |
| Valores negativos en `Quantity`/`Price` | Filtrar > 0. |
| Outliers de cantidad | Conservar pero transformar (log). |
| Dataset dominado por UK | Si usamos país, agruparlo. |

---

## Preguntas de Reflexión

1. ¿Por qué no podemos imputar los `CustomerID` faltantes con la moda
   (el cliente más frecuente)?
2. ¿Qué problemas traería *no* eliminar las cancelaciones antes del RFM?
3. Si el negocio te pidiera segmentar **clientes mayoristas** específicamente,
   ¿qué variables nuevas considerarías?
4. ¿Qué pasaría si entrenáramos K-Means usando `Quantity` directamente,
   sin transformar los outliers?

> **Próximo paso:** ``02_data_cleaning.ipynb`` — aplicamos las decisiones
> que tomamos aquí.
"""
        ),
    ]
    write_notebook("01_data_understanding.ipynb", cells)


# ===========================================================================
# 02 — Data Cleaning
# ===========================================================================
def build_02():
    cells = [
        md(
            """\
# 02 · Limpieza de Datos

> **Objetivo:** convertir el dataset crudo en un conjunto consistente
> y confiable para el feature engineering.

## ¿Qué vamos a hacer?

1. Eliminar filas sin `CustomerID`.
2. Eliminar cancelaciones.
3. Filtrar `Quantity > 0` y `Price > 0`.
4. Eliminar duplicados exactos.
5. Calcular `LineTotal = Quantity * Price`.
6. Guardar el resultado en `data/interim/transacciones_limpias.parquet`.

## Concepto teórico: las tres opciones de limpieza

Frente a un dato problemático tienes **tres** opciones, y elegir bien
es lo que separa a un científico de datos junior de uno senior:

| Opción | Cuándo aplica | Riesgos |
|---|---|---|
| **Eliminar** | Cuando el dato está irrecuperablemente roto o no se puede atribuir. | Sesgo si los faltantes no son aleatorios. |
| **Imputar** | Cuando hay un valor "por defecto" razonable (mediana, moda). | Aplanar la varianza, inventar señal. |
| **Transformar** | Cuando el dato es real pero "extremo" (outliers, sesgo). | Distorsionar la interpretación. |

> **Regla de oro:** documenta CADA decisión. En entrevistas y auditorías
> te van a preguntar el *por qué*, no el *qué*.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd

from segmentacion.data.loader import load_raw_transactions
from segmentacion.features.rfm import clean_transactions
from segmentacion.config import CLEAN_DATA_FILE

pd.set_option("display.max_columns", 30)
"""
        ),
        md("## 1. Cargar el dataset crudo"),
        code(
            """\
df = load_raw_transactions()
print(f"Filas iniciales: {len(df):,}")
"""
        ),
        md(
            """\
## 2. Aplicar las reglas de limpieza

La función ``clean_transactions`` aplica las cinco reglas en orden y
devuelve el dataframe limpio. Vamos a verla en acción y luego desglosarla
manualmente para entender cada paso.
"""
        ),
        code(
            """\
df_clean = clean_transactions(df)
print(f"Filas después de limpieza: {len(df_clean):,}")
df_clean.head()
"""
        ),
        md(
            """\
## 3. Desglose paso a paso (didáctico)

Para entender *qué* hace `clean_transactions`, repliquemos sus pasos.
"""
        ),
        code(
            """\
n0 = len(df)
print(f"0. Inicial: {n0:,}")

# Paso 1: eliminar sin CustomerID
df_step = df.dropna(subset=["CustomerID"]).copy()
print(f"1. Sin CustomerID NaN: {len(df_step):,} (-{n0 - len(df_step):,})")

# Paso 2: eliminar cancelaciones
df_step["Invoice"] = df_step["Invoice"].astype(str)
n_prev = len(df_step)
df_step = df_step[~df_step["Invoice"].str.startswith("C")]
print(f"2. Sin cancelaciones: {len(df_step):,} (-{n_prev - len(df_step):,})")

# Paso 3: cantidades y precios positivos
n_prev = len(df_step)
df_step = df_step[(df_step["Quantity"] > 0) & (df_step["Price"] > 0)]
print(f"3. Quantity y Price > 0: {len(df_step):,} (-{n_prev - len(df_step):,})")

# Paso 4: duplicados
n_prev = len(df_step)
df_step = df_step.drop_duplicates()
print(f"4. Sin duplicados exactos: {len(df_step):,} (-{n_prev - len(df_step):,})")
"""
        ),
        md(
            """\
> **Tip:** mantener un log con el conteo en cada paso es invaluable.
> Si mañana cambia el dataset y los números bajan demasiado, sabes
> exactamente en qué paso revisar.

## 4. Validación post-limpieza
"""
        ),
        code(
            """\
checks = {
    "CustomerID nulos": df_clean["CustomerID"].isna().sum(),
    "Cancelaciones (C*)": df_clean["Invoice"].astype(str).str.startswith("C").sum(),
    "Quantity <= 0": (df_clean["Quantity"] <= 0).sum(),
    "Price <= 0": (df_clean["Price"] <= 0).sum(),
    "Duplicados": df_clean.duplicated().sum(),
}
for k, v in checks.items():
    status = "OK" if v == 0 else "FALLA"
    print(f"  [{status}] {k}: {v}")
"""
        ),
        md(
            """\
## 5. Detección de outliers (sin eliminarlos)

A diferencia de las cuatro reglas anteriores, los outliers en `Quantity`
y `Monetary` **no los eliminamos**. Razones:

1. Pueden ser **clientes mayoristas legítimos**, que son un segmento de negocio importante.
2. La transformación logarítmica que aplicaremos en los pipelines reduce su influencia.
3. Eliminar outliers a ciegas puede esconder el segmento más rentable.

Veamos su magnitud:
"""
        ),
        code(
            """\
df_clean["LineTotal"] = df_clean["Quantity"] * df_clean["Price"]

p99 = df_clean[["Quantity", "Price", "LineTotal"]].quantile([0.5, 0.95, 0.99, 1.0])
print("Percentiles de variables continuas:")
print(p99)
"""
        ),
        md(
            """\
> **Observa** la diferencia entre la mediana y el máximo. Eso es un
> long-tail clásico.

## 6. Guardar el resultado

Lo guardamos en `parquet` (más compacto y rápido que CSV).
"""
        ),
        code(
            """\
df_clean.to_parquet(CLEAN_DATA_FILE, index=False)
print(f"Guardado en: {CLEAN_DATA_FILE}")
print(f"Tamaño: {CLEAN_DATA_FILE.stat().st_size / 1e6:.2f} MB")
"""
        ),
        md(
            """\
## Resumen

| Decisión | Razón |
|---|---|
| Eliminar `CustomerID` NaN | No se pueden atribuir; imputar inventaría. |
| Eliminar cancelaciones | No son compras netas. |
| Filtrar `Quantity` y `Price` > 0 | Errores de captura. |
| Eliminar duplicados exactos | Errores de carga. |
| **Conservar outliers** | Pueden ser mayoristas → segmento valioso. |

---

## Preguntas de Reflexión

1. Si hicieras el RFM **restando** las cancelaciones (en lugar de
   eliminarlas), ¿qué cambios verías en el segmento "Champions"?
2. ¿Por qué `parquet` es preferible a `csv` para datos numéricos grandes?
3. ¿Qué pasaría si un cliente mayorista hace una sola gran compra?
   ¿En qué cluster terminaría con K-Means? ¿Y con DBSCAN?

> **Próximo paso:** ``03_feature_engineering.ipynb`` — convertir las
> transacciones en perfiles RFM por cliente.
"""
        ),
    ]
    write_notebook("02_data_cleaning.ipynb", cells)


# ===========================================================================
# 03 — Feature Engineering
# ===========================================================================
def build_03():
    cells = [
        md(
            """\
# 03 · Feature Engineering

> **Objetivo:** convertir un dataset *transaccional* en un dataset *por cliente*,
> donde cada fila representa el comportamiento agregado de un cliente.

## ¿Qué vamos a hacer?

1. Definir el **snapshot date** (fecha de referencia).
2. Calcular RFM clásico: **Recency**, **Frequency**, **Monetary**.
3. Extender RFM con: **AvgTicket**, **ProductDiversity**, **AvgQuantity**.
4. Validar la calidad de las features.
5. Guardar el dataset por cliente.

## Concepto teórico: ¿por qué RFM?

RFM nació en los 90s en marketing directo. La intuición:

- *Si compraste recientemente, es más probable que vuelvas a comprar.*
- *Si compras seguido, ya eres un cliente fiel.*
- *Si gastas mucho, eres valioso para el negocio.*

Décadas de evidencia empírica respaldan que estas tres dimensiones,
incluso sin más features, ya capturan la mayor parte de la variabilidad
de comportamiento de clientes en retail.

> **Tip didáctico:** RFM es el "Hello World" de la segmentación.
> Empieza siempre por aquí antes de meter features más sofisticadas.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd
import numpy as np

from segmentacion.features.rfm import build_customer_features, build_rfm
from segmentacion.config import CLEAN_DATA_FILE, RFM_DATA_FILE, FEATURES_DATA_FILE
"""
        ),
        md("## 1. Cargar las transacciones limpias"),
        code(
            """\
df = pd.read_parquet(CLEAN_DATA_FILE)
print(f"Transacciones: {len(df):,}")
print(f"Clientes únicos: {df['CustomerID'].nunique():,}")
"""
        ),
        md(
            """\
## 2. Snapshot date

Es la fecha "hoy" desde la cual medimos `Recency`. Usamos `max + 1 día`
para evitar que algún cliente tenga `Recency = 0`.
"""
        ),
        code(
            """\
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date.date()}")
"""
        ),
        md(
            """\
## 3. RFM clásico

Calculamos las tres métricas con un solo `groupby`:
"""
        ),
        code(
            """\
rfm = build_rfm(df, snapshot_date=snapshot_date)
rfm.head()
"""
        ),
        md(
            """\
**Lectura del output:**

- `Recency` en días.
- `Frequency` = nº de facturas distintas.
- `Monetary` = gasto total acumulado.

Un cliente con `Recency=2, Frequency=10, Monetary=5000` es muy valioso.
Uno con `Recency=300, Frequency=1, Monetary=20` es probablemente perdido.

## 4. Distribución de RFM
"""
        ),
        code(
            """\
rfm.describe().T
"""
        ),
        md(
            """\
> **Observa el sesgo:** la mediana de `Monetary` es mucho menor que la
> media. Eso confirma una distribución long-tail (pocos clientes muy
> grandes inflan la media). Tendremos que aplicar `log1p` antes del
> escalado.

## 5. Features extendidas

Tres variables adicionales que ayudan al clustering a distinguir matices:

- **AvgTicket** = `Monetary / Frequency`. Distingue clientes de muchas
  compras pequeñas vs. pocas compras grandes.
- **ProductDiversity** = nº de StockCode distintos. Un cliente con
  diversidad 1 compra siempre lo mismo (¿suministro?, ¿hábito?).
- **AvgQuantity** = cantidad promedio de items por factura. Alta
  podría indicar comportamiento mayorista.
"""
        ),
        code(
            """\
features = build_customer_features(df, snapshot_date=snapshot_date)
features.head()
"""
        ),
        md("## 6. Estadísticas y correlaciones"),
        code(
            """\
features.describe().T
"""
        ),
        code(
            """\
features.corr().round(2)
"""
        ),
        md(
            """\
> **Esperado:** `Frequency`, `Monetary` y `ProductDiversity` están
> fuertemente correlacionadas. Tiene sentido: comprar más veces implica
> gastar más y probar más productos. La correlación es información, no
> un problema. Lo importante es que K-Means y DBSCAN no requieren
> independencia (a diferencia de la regresión).

## 7. Validación de calidad
"""
        ),
        code(
            """\
checks = {
    "Filas con NaN": features.isna().any(axis=1).sum(),
    "Recency negativa": (features["Recency"] < 0).sum(),
    "Frequency cero": (features["Frequency"] <= 0).sum(),
    "Monetary cero o negativa": (features["Monetary"] <= 0).sum(),
}
for k, v in checks.items():
    status = "OK" if v == 0 else "REVISAR"
    print(f"  [{status}] {k}: {v}")
"""
        ),
        md(
            """\
## 8. Top 10 clientes por valor
"""
        ),
        code(
            """\
features.nlargest(10, "Monetary")
"""
        ),
        md(
            """\
> Estos clientes son los Champions / VIPs. Es probable que terminen en
> un cluster aparte (o sean marcados como outliers por DBSCAN).

## 9. Guardar features
"""
        ),
        code(
            """\
rfm.to_parquet(RFM_DATA_FILE)
features.to_parquet(FEATURES_DATA_FILE)
print(f"RFM básico:    {RFM_DATA_FILE}")
print(f"Features ext.: {FEATURES_DATA_FILE}")
"""
        ),
        md(
            """\
## Resumen

| Feature | Calculo | Insight de negocio |
|---|---|---|
| Recency | Días desde última compra | Bajo = activo |
| Frequency | Nº de facturas | Alto = fiel |
| Monetary | Gasto total | Alto = valioso |
| AvgTicket | Monetary / Frequency | Distingue volumen vs. ticket |
| ProductDiversity | Nº productos únicos | Alto = explorador |
| AvgQuantity | Items promedio por factura | Alto = mayorista? |

---

## Preguntas de Reflexión

1. ¿Por qué `Frequency` y `Monetary` tienen correlación alta? ¿Eso
   significa que una de las dos es redundante? Justifica.
2. Si calcularas `Recency` con `snapshot_date = '2010-01-01'` (en medio
   del dataset), ¿qué problemas tendrías?
3. ¿Qué otra feature de comportamiento se te ocurre que no esté aquí?
4. Para un negocio **B2B** (suministros industriales), ¿modificarías
   alguna definición de RFM?

> **Próximo paso:** ``04_exploratory_data_analysis.ipynb`` — visualizar
> las features para tomar decisiones de modelado.
"""
        ),
    ]
    write_notebook("03_feature_engineering.ipynb", cells)


# ===========================================================================
# 04 — EDA
# ===========================================================================
def build_04():
    cells = [
        md(
            """\
# 04 · Análisis Exploratorio (EDA)

> **Objetivo:** entender la *forma* de las features antes de modelar.
> El EDA cambia las decisiones de preprocesamiento.

## ¿Qué vamos a hacer?

1. Visualizar distribuciones de cada feature.
2. Identificar sesgo y outliers.
3. Mostrar el efecto de la transformación `log1p`.
4. Visualizar correlaciones.
5. Hacer un primer scatter plot (Recency vs Monetary) con tamaño = Frequency.

## ¿Por qué importa el EDA antes del clustering?

Porque las decisiones de preprocesamiento dependen de cómo se ven los datos:

- ¿Distribución sesgada? → log + escalar.
- ¿Outliers extremos? → robust scaler o transformaciones.
- ¿Variables en escalas muy distintas? → escalar (StandardScaler).
- ¿Variables muy correlacionadas? → considerar PCA.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from segmentacion.config import FEATURES_DATA_FILE
from segmentacion.visualization.plots import plot_distributions

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
"""
        ),
        code(
            """\
features = pd.read_parquet(FEATURES_DATA_FILE)
print(f"Clientes: {len(features):,}")
features.describe().T
"""
        ),
        md(
            """\
## 1. Distribuciones (histograma + boxplot)

La función `plot_distributions` del módulo de visualización dibuja
ambas gráficas lado a lado para cada variable.
"""
        ),
        code(
            """\
fig = plot_distributions(features, columns=["Recency", "Frequency", "Monetary"])
plt.show()
"""
        ),
        md(
            """\
**Lo que deberías ver:**

- **Recency**: bimodal o uniforme. Hay clientes recientes y otros muy antiguos.
- **Frequency**: extremadamente sesgada a la derecha. La mayoría compró pocas
  veces; unos pocos compraron decenas o cientos.
- **Monetary**: peor que Frequency, long-tail brutal.

> **Errores comunes:**
>
> - Aplicar K-Means sobre `Monetary` sin transformar produce clusters
>   degenerados: un cluster diminuto con los mayoristas y otro gigante
>   con todo lo demás.
> - "Eliminar outliers" usando IQR rebana segmentos legítimos.

## 2. Efecto de log1p

`log1p(x) = log(1 + x)` comprime la cola larga sin hacer estallar el log
en los ceros. Es la transformación canónica para variables long-tail.
"""
        ),
        code(
            """\
log_features = features[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
log_features.columns = [f"log_{c}" for c in log_features.columns]

fig = plot_distributions(log_features, columns=log_features.columns)
plt.show()
"""
        ),
        md(
            """\
**Compara** estos histogramas con los anteriores. Las distribuciones
ahora se parecen más a campanas. K-Means y DBSCAN funcionan mucho mejor
con este input.

## 3. Correlaciones (heatmap)
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    features.corr(),
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    square=True,
    ax=ax,
)
ax.set_title("Correlaciones entre features")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
> **Observa:** `Frequency` ↔ `Monetary` ↔ `ProductDiversity` están
> muy correlacionadas. No es un problema para clustering, pero sí
> indica que podrías reducir dimensionalidad con PCA.

## 4. Scatter Recency vs Monetary, escala log

Una primera visualización del espacio de clientes:
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    features["Recency"],
    features["Monetary"],
    s=features["Frequency"] * 2,
    alpha=0.4,
    c=features["Frequency"],
    cmap="viridis",
)
ax.set_xlabel("Recency (días desde última compra)")
ax.set_ylabel("Monetary (GBP, log)")
ax.set_yscale("log")
ax.set_title("Espacio de clientes: Recency vs Monetary (tamaño y color = Frequency)")
plt.colorbar(sc, label="Frequency")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
**Lectura:**

- Los clientes con **Recency baja + Monetary alta + Frequency alta** son
  los Champions (esquina inferior derecha del scatter).
- Los clientes con **Recency alta + Monetary baja** son inactivos.
- Hay puntos aislados arriba a la derecha → outliers / mayoristas.

## 5. Pairplot rápido (cuidado con datasets grandes)
"""
        ),
        code(
            """\
sample = features.sample(min(2000, len(features)), random_state=42)
sns.pairplot(np.log1p(sample[["Recency", "Frequency", "Monetary", "AvgTicket"]]),
             plot_kws={"alpha": 0.3, "s": 8}, height=2.2)
plt.suptitle("Pairplot (log) — muestra de 2000 clientes", y=1.02)
plt.show()
"""
        ),
        md(
            """\
## Resumen

| Hallazgo | Implicación |
|---|---|
| Long-tail en Monetary y Frequency | Transformar con log1p antes de escalar. |
| Correlación alta RFM | OK para clustering; PCA opcional. |
| Outliers visibles | Conservarlos (mayoristas potenciales). |
| Distribución bimodal en Recency | Probable separación natural en clusters. |

---

## Preguntas de Reflexión

1. ¿Por qué `log1p(x)` es preferible a `log(x)` cuando hay valores cercanos a 0?
2. Si una feature tiene varianza casi cero, ¿qué efecto tiene en K-Means?
3. ¿Qué información se pierde al aplicar `log`? ¿Es importante esa pérdida?
4. ¿Detectaste algún subgrupo a simple vista en el scatter? ¿Cuántos clusters
   "esperarías" que K-Means encuentre?

> **Próximo paso:** ``05_preprocessing_pipelines.ipynb`` — formalizar
> las transformaciones en pipelines reproducibles.
"""
        ),
    ]
    write_notebook("04_exploratory_data_analysis.ipynb", cells)


# ===========================================================================
# 05 — Preprocessing Pipelines
# ===========================================================================
def build_05():
    cells = [
        md(
            """\
# 05 · Pipelines de Preprocesamiento

> **Objetivo:** dejar listo un pipeline de scikit-learn que transforme
> el dataset por cliente en una matriz numérica lista para clusterizar.

## ¿Qué vamos a hacer?

1. Construir un pipeline numérico (imputación → log → escalado).
2. Construir un pipeline categórico (imputación → one-hot).
3. Combinarlos con `ColumnTransformer`.
4. Agregar PCA al final como opción.
5. Comparar el efecto de **no escalar** vs escalar.

## Concepto teórico: ¿por qué pipelines?

Un `Pipeline` de scikit-learn encadena transformaciones. Las ventajas son:

1. **Reproducibilidad**: el mismo objeto aplica las mismas transformaciones.
2. **Sin data leakage**: en `fit_transform(train)` el escalador aprende
   parámetros del train; al aplicar a test usa esos mismos parámetros.
3. **Serialización**: puedes guardar el pipeline entero en un `.joblib`
   y cargarlo en producción (ej. la app de Streamlit).

`ColumnTransformer` permite aplicar diferentes pipelines a diferentes
columnas: numéricas a uno, categóricas a otro, etc.

## ¿Por qué escalar es CRÍTICO para K-Means y DBSCAN?

Ambos algoritmos usan **distancia euclidiana**. Si tienes:

- `Frequency` ∈ [1, 100]
- `Monetary` ∈ [0, 200000]

entonces la distancia entre dos clientes está dominada casi 100% por
`Monetary`. `Frequency` se vuelve invisible. Escalar pone todas las
variables en el mismo rango y deja que el algoritmo "vea" todas
las dimensiones.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd
import numpy as np

from segmentacion.config import FEATURES_DATA_FILE, RFM_FEATURES, EXTENDED_NUMERIC_FEATURES
from segmentacion.features.preprocessing import build_preprocessing_pipeline
"""
        ),
        code(
            """\
features = pd.read_parquet(FEATURES_DATA_FILE)
features.head()
"""
        ),
        md(
            """\
## 1. Pipeline RFM básico (numérico únicamente)

Variables: `Recency`, `Frequency`, `Monetary`. Imputación → log → escalado.
"""
        ),
        code(
            """\
pipe_rfm = build_preprocessing_pipeline(
    numeric_features=RFM_FEATURES,
    categorical_features=None,
    use_log=True,
    n_components_pca=None,
)
X_rfm = pipe_rfm.fit_transform(features[RFM_FEATURES])
print(f"Shape post-pipeline: {X_rfm.shape}")
print(f"Media:  {X_rfm.mean(axis=0).round(3)}")
print(f"Stddev: {X_rfm.std(axis=0).round(3)}")
"""
        ),
        md(
            """\
> **Tras `StandardScaler`** la media es ~0 y la desviación ~1. Eso es
> exactamente lo que queremos para K-Means.

## 2. Pipeline extendido (RFM + features adicionales)
"""
        ),
        code(
            """\
pipe_ext = build_preprocessing_pipeline(
    numeric_features=EXTENDED_NUMERIC_FEATURES,
    use_log=True,
)
X_ext = pipe_ext.fit_transform(features[EXTENDED_NUMERIC_FEATURES])
print(f"Shape: {X_ext.shape}")
"""
        ),
        md(
            """\
## 3. Pipeline con PCA (para visualización 2D)

PCA proyecta los datos a un subespacio de menor dimensión preservando
la mayor varianza posible. Útil para visualizar y, opcionalmente,
para reducir colinealidad antes de clusterizar.
"""
        ),
        code(
            """\
pipe_pca = build_preprocessing_pipeline(
    numeric_features=EXTENDED_NUMERIC_FEATURES,
    use_log=True,
    n_components_pca=2,
)
X_pca = pipe_pca.fit_transform(features[EXTENDED_NUMERIC_FEATURES])
print(f"Shape post-PCA: {X_pca.shape}")

pca_step = pipe_pca.named_steps["pca"]
print(f"Varianza explicada por componente: {pca_step.explained_variance_ratio_.round(3)}")
print(f"Total: {pca_step.explained_variance_ratio_.sum():.2%}")
"""
        ),
        md(
            """\
> Si las dos primeras componentes ya capturan >80% de varianza,
> visualizar en 2D es razonable. Si no, considera 3D o usa t-SNE/UMAP.

## 4. Pipeline con variables categóricas

Si quisiéramos incluir el `Country` agrupado, podríamos hacerlo así.
(Ejemplo ilustrativo: en este proyecto principal usaremos solo numéricas.)
"""
        ),
        code(
            """\
# Ejemplo: agrupar países en categorías
features_cat = features.copy()
# Necesitaríamos joinar con la tabla original para obtener country.
# Aquí solo mostramos la estructura del pipeline.

pipe_mix = build_preprocessing_pipeline(
    numeric_features=RFM_FEATURES,
    categorical_features=None,  # cambia por ['CountryGroup'] si lo agregas
    use_log=True,
)
print(pipe_mix)
"""
        ),
        md(
            """\
## 5. Demostración del efecto del escalado

Veamos qué pasa con K-Means **sin** escalar vs **con** escalar.
"""
        ),
        code(
            """\
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_raw = features[RFM_FEATURES].values
labels_raw = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_raw)
sil_raw = silhouette_score(X_raw, labels_raw)

labels_scaled = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_rfm)
sil_scaled = silhouette_score(X_rfm, labels_scaled)

print(f"Sin escalar:      Silhouette = {sil_raw:.3f}")
print(f"Con log+escalar:  Silhouette = {sil_scaled:.3f}")
"""
        ),
        md(
            """\
> **Tip de evaluación:** el Silhouette suele subir bastante (a veces 2-3x)
> tras escalar correctamente. Es la prueba más rápida de que el pipeline
> está haciendo su trabajo.

## 6. Cuándo usar cada escalador

| Escalador | Cuándo |
|---|---|
| `StandardScaler` | Default. Centra en 0 y escala a varianza 1. |
| `MinMaxScaler` | Cuando necesitas valores en [0, 1] (ej. visualizaciones, NN). |
| `RobustScaler` | Cuando hay outliers que no quieres transformar (usa mediana e IQR). |
| `MaxAbsScaler` | Cuando los datos son sparse y quieres mantener el cero. |

## 7. Guardar el pipeline para uso posterior
"""
        ),
        code(
            """\
import joblib
from segmentacion.config import PIPELINE_FILE

joblib.dump(pipe_ext, PIPELINE_FILE)
print(f"Pipeline guardado en: {PIPELINE_FILE}")
"""
        ),
        md(
            """\
## Resumen

- `Pipeline` → secuencia ordenada de transformaciones.
- `ColumnTransformer` → diferentes transformaciones por columna.
- Para RFM: imputación → `log1p` → `StandardScaler`.
- Escalar es **crítico** para K-Means y DBSCAN.

---

## Preguntas de Reflexión

1. ¿Por qué un `MinMaxScaler` puede ser mala idea cuando hay outliers?
2. ¿Qué pasaría si entrenaras el `StandardScaler` sobre `train + test`?
3. ¿En qué situación un PCA *empeora* el clustering?
4. Si quisieras dar más peso a `Monetary` en el clustering, ¿cómo lo harías
   sin romper el pipeline?

> **Próximo paso:** ``06_clustering_kmeans.ipynb`` — al fin clusterizamos.
"""
        ),
    ]
    write_notebook("05_preprocessing_pipelines.ipynb", cells)


# ===========================================================================
# 06 — K-Means
# ===========================================================================
def build_06():
    cells = [
        md(
            """\
# 06 · Clustering con K-Means

> **Objetivo:** entrenar K-Means, elegir el k óptimo, y persistir el modelo.

## ¿Qué vamos a hacer?

1. Recapitular cómo funciona K-Means.
2. Aplicar el método del codo y la silueta para elegir `k`.
3. Entrenar el modelo final.
4. Visualizar los clusters en 2D (PCA).
5. Guardar el modelo.

## Concepto teórico

K-Means itera dos pasos:

1. **Asignación**: cada punto se asigna al **centroide** más cercano.
2. **Actualización**: cada centroide se mueve al centro de sus puntos.

Repite hasta que los centroides dejan de moverse. Minimiza la **inertia**:

$$\\text{Inertia} = \\sum_{i=1}^{n} \\min_{c \\in C} \\|x_i - c\\|^2$$

### Fortalezas
- Rápido (escalable a datasets grandes).
- Determinístico con `random_state`.
- Fácil de interpretar (centroides = "cliente promedio" del cluster).

### Limitaciones
- Necesitas elegir `k` manualmente.
- Asume clusters **esféricos** y de tamaño **similar**.
- Sensible a outliers (los centroides se ven jalados).
- Sensible a la escala → siempre escalar antes.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from segmentacion.config import (
    FEATURES_DATA_FILE,
    EXTENDED_NUMERIC_FEATURES,
    KMEANS_K_RANGE,
    KMEANS_MODEL_FILE,
    PIPELINE_FILE,
    SEGMENTS_FILE,
    RANDOM_STATE,
)
from segmentacion.features.preprocessing import build_preprocessing_pipeline
from segmentacion.models.clustering import find_optimal_k, fit_kmeans
from segmentacion.visualization.plots import plot_clusters_2d, plot_elbow_and_silhouette
from sklearn.decomposition import PCA
"""
        ),
        md("## 1. Cargar features y aplicar pipeline"),
        code(
            """\
features = pd.read_parquet(FEATURES_DATA_FILE)
print(f"Clientes: {len(features):,}")

pipeline = build_preprocessing_pipeline(
    numeric_features=EXTENDED_NUMERIC_FEATURES,
    use_log=True,
)
X = pipeline.fit_transform(features[EXTENDED_NUMERIC_FEATURES])
print(f"Matriz de features: {X.shape}")
"""
        ),
        md(
            """\
## 2. Búsqueda del K óptimo

Probamos `k` de 2 a 10 y graficamos inertia (codo) y silueta.
"""
        ),
        code(
            """\
metrics = find_optimal_k(X, k_range=KMEANS_K_RANGE, random_state=RANDOM_STATE)
metrics
"""
        ),
        code(
            """\
fig = plot_elbow_and_silhouette(metrics)
plt.show()
"""
        ),
        md(
            """\
**Cómo leer estas curvas:**

- **Codo (inertia):** busca el "codo" donde la pendiente se aplana.
  Pasar de k=3 a k=4 te da una mejora grande; pasar de k=8 a k=9 te da
  una mejora marginal.
- **Silueta:** picos altos indican clusters bien definidos. Si k=4 tiene
  silueta=0.45 y k=5 tiene 0.48, la diferencia puede no ser significativa.

> **Tip:** no elijas el k que maximiza la silueta a ciegas. **K=2** suele
> ganar siempre porque dividir el dataset en dos mitades grandes da
> alta silueta, pero rara vez tiene sentido de negocio.

## 3. Davies-Bouldin y Calinski-Harabasz
"""
        ),
        code(
            """\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(metrics.index, metrics["davies_bouldin"], "o-", color="purple")
axes[0].set_title("Davies-Bouldin (más BAJO mejor)")
axes[0].set_xlabel("k"); axes[0].set_ylabel("DB Index")
axes[0].grid(True, alpha=0.3)

axes[1].plot(metrics.index, metrics["calinski_harabasz"], "o-", color="teal")
axes[1].set_title("Calinski-Harabasz (más ALTO mejor)")
axes[1].set_xlabel("k"); axes[1].set_ylabel("CH Index")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 4. Decisión de k

Combinando las cuatro métricas + sentido de negocio:

- Para Online Retail II, **k=4** suele ser un compromiso razonable.
- Da segmentos accionables: Champions, Leales, En Riesgo, Perdidos.
- Más k aumenta la complejidad de la estrategia de marketing sin
  ganar mucho en homogeneidad.

> **Tip:** la decisión final debe consensuarse con el equipo de marketing.
> El "mejor k" depende de cuántos segmentos el negocio puede activar.

## 5. Entrenar el modelo final con k = 4
"""
        ),
        code(
            """\
K_FINAL = 4
kmeans = fit_kmeans(X, n_clusters=K_FINAL, random_state=RANDOM_STATE)

features["cluster_kmeans"] = kmeans.labels_
print("Distribución de clientes por cluster:")
print(features["cluster_kmeans"].value_counts().sort_index())
"""
        ),
        md(
            """\
## 6. Visualización 2D con PCA
"""
        ),
        code(
            """\
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X)
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")

fig = plot_clusters_2d(X_2d, kmeans.labels_, title=f"K-Means · k={K_FINAL}")
plt.show()
"""
        ),
        md(
            """\
## 7. Centroides en escala original

Los centroides en el espacio escalado no son interpretables. Para entender
"cómo es" cada cluster, calculamos el promedio de las features originales
por cluster.
"""
        ),
        code(
            """\
profile_kmeans = features.groupby("cluster_kmeans")[EXTENDED_NUMERIC_FEATURES].mean().round(2)
profile_kmeans["n_customers"] = features["cluster_kmeans"].value_counts().sort_index()
profile_kmeans
"""
        ),
        md(
            """\
> Interpreta cada cluster: ¿quién tiene la Recency más baja?
> ¿Quién tiene la Monetary más alta? Esos son tus Champions.

## 8. Guardar el modelo y los segmentos
"""
        ),
        code(
            """\
joblib.dump(kmeans, KMEANS_MODEL_FILE)
joblib.dump(pipeline, PIPELINE_FILE)
features.reset_index().to_parquet(SEGMENTS_FILE, index=False)
print(f"Modelo:    {KMEANS_MODEL_FILE}")
print(f"Pipeline:  {PIPELINE_FILE}")
print(f"Segmentos: {SEGMENTS_FILE}")
"""
        ),
        md(
            """\
## Resumen

- K-Means minimiza la inertia → busca clusters esféricos.
- El **codo** y la **silueta** ayudan a elegir `k`, pero no sustituyen
  el juicio de negocio.
- Los centroides son interpretables si los devuelves a la escala original.

---

## Preguntas de Reflexión

1. ¿Por qué la silueta para `k=2` es casi siempre la más alta? ¿Por qué
   raramente es la decisión correcta?
2. Si `random_state` no estuviera fijado, ¿qué tan estables serían los clusters?
3. ¿Qué cluster esperarías que sea más sensible a una promoción?
4. ¿Qué pasaría si tuvieras un cliente con `Frequency=200` y `Monetary=500000`?
   ¿En qué cluster terminaría? ¿Distorsionaría el centroide?

> **Próximo paso:** ``07_clustering_dbscan.ipynb`` — un enfoque distinto
> que detecta outliers automáticamente.
"""
        ),
    ]
    write_notebook("06_clustering_kmeans.ipynb", cells)


# ===========================================================================
# 07 — DBSCAN
# ===========================================================================
def build_07():
    cells = [
        md(
            """\
# 07 · Clustering con DBSCAN

> **Objetivo:** aplicar DBSCAN, entender sus parámetros, y comparar
> sus resultados con K-Means.

## ¿Qué vamos a hacer?

1. Repasar el algoritmo y sus dos hiperparámetros.
2. Elegir `eps` con la curva k-distance.
3. Probar diferentes configuraciones.
4. Visualizar los clusters y los outliers.

## Concepto teórico

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
agrupa puntos basándose en **densidad local**, no en distancia a un centroide.

### Definiciones clave

- **eps (ε)**: radio del vecindario.
- **min_samples**: nº mínimo de puntos en el vecindario para considerar
  un punto como **núcleo**.
- **Punto núcleo**: tiene ≥ `min_samples` vecinos dentro de ε.
- **Punto frontera**: está dentro del vecindario de un núcleo, pero
  no tiene él mismo ≥ `min_samples` vecinos.
- **Ruido (noise)**: ni núcleo ni frontera. Etiqueta `-1`.

### Diferencias clave con K-Means

| K-Means | DBSCAN |
|---|---|
| Necesita `k` | No, descubre el nº de clusters. |
| Asume clusters esféricos | Encuentra **formas arbitrarias**. |
| Asigna TODO punto | Marca outliers como `-1`. |
| Sensible a outliers | Robusto (los aísla). |
| Escala bien a alta dimensión | Sufre con muchas dimensiones (curse of dimensionality). |

> **Tip:** DBSCAN brilla cuando esperas que existan *outliers genuinos*
> (mayoristas, fraudes) que no quieres forzar a un cluster.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from segmentacion.config import (
    FEATURES_DATA_FILE,
    EXTENDED_NUMERIC_FEATURES,
    DBSCAN_MODEL_FILE,
    PIPELINE_FILE,
    RANDOM_STATE,
)
from segmentacion.features.preprocessing import build_preprocessing_pipeline
from segmentacion.models.clustering import evaluate_clustering, fit_dbscan
from segmentacion.visualization.plots import plot_clusters_2d, plot_k_distance
"""
        ),
        md("## 1. Preparar features"),
        code(
            """\
features = pd.read_parquet(FEATURES_DATA_FILE)
pipeline = build_preprocessing_pipeline(
    numeric_features=EXTENDED_NUMERIC_FEATURES,
    use_log=True,
)
X = pipeline.fit_transform(features[EXTENDED_NUMERIC_FEATURES])
print(f"Matriz: {X.shape}")
"""
        ),
        md(
            """\
## 2. Elegir `eps` con la curva k-distance

Heurística clásica (Ester et al., 1996):

1. Calcula la distancia de cada punto a su `min_samples`-ésimo vecino.
2. Ordena esas distancias de menor a mayor.
3. Grafica → busca el "codo".
4. El valor del codo en el eje y es un buen `eps` inicial.
"""
        ),
        code(
            """\
MIN_SAMPLES = 2 * X.shape[1]  # heurística: 2 * nº de features
print(f"min_samples sugerido: {MIN_SAMPLES}")

fig = plot_k_distance(X, k=MIN_SAMPLES)
plt.show()
"""
        ),
        md(
            """\
> **Tip:** el codo está donde la curva pasa de plana a empinada.
> Para este dataset suele estar en `eps ≈ 0.5 - 1.0`. Vamos a probar varios.

## 3. Comparar varias combinaciones de hiperparámetros
"""
        ),
        code(
            """\
configs = [
    (0.3, MIN_SAMPLES),
    (0.5, MIN_SAMPLES),
    (0.7, MIN_SAMPLES),
    (1.0, MIN_SAMPLES),
    (0.5, MIN_SAMPLES // 2),
    (0.5, MIN_SAMPLES * 2),
]

rows = []
for eps, ms in configs:
    model = fit_dbscan(X, eps=eps, min_samples=ms)
    metrics = evaluate_clustering(X, model.labels_)
    rows.append({"eps": eps, "min_samples": ms, **metrics})

results = pd.DataFrame(rows)
results
"""
        ),
        md(
            """\
**Cómo interpretar la tabla:**

- `n_clusters` = clusters reales (sin contar el -1 de ruido).
- `n_noise` = puntos clasificados como outliers.
- Una buena configuración:
  - tiene `n_clusters >= 2` (si es 1, eps está demasiado grande).
  - tiene `n_noise` razonable (~5-15%; si es 50%+, eps está demasiado
    pequeño).
  - tiene silueta positiva.

> **Errores comunes:**
>
> - **eps muy pequeño** → casi todo es ruido.
> - **eps muy grande** → un solo cluster gigante.
> - **min_samples muy alto** → muchos puntos quedan sin clasificar.

## 4. Configuración final
"""
        ),
        code(
            """\
EPS_FINAL = 0.5
MS_FINAL = MIN_SAMPLES
dbscan = fit_dbscan(X, eps=EPS_FINAL, min_samples=MS_FINAL)
features["cluster_dbscan"] = dbscan.labels_

print("Distribución de clientes (incluye -1 = ruido):")
print(features["cluster_dbscan"].value_counts().sort_index())
"""
        ),
        md("## 5. Visualización 2D"),
        code(
            """\
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X)

fig = plot_clusters_2d(
    X_2d,
    dbscan.labels_,
    title=f"DBSCAN · eps={EPS_FINAL}, min_samples={MS_FINAL}",
)
plt.show()
"""
        ),
        md(
            """\
> Los puntos grises son ruido. Observa cómo DBSCAN los separa
> automáticamente, mientras que K-Means los habría asignado a algún
> cluster jalando los centroides.

## 6. ¿Quiénes son los outliers?
"""
        ),
        code(
            """\
outliers = features[features["cluster_dbscan"] == -1]
print(f"Outliers detectados: {len(outliers)}")
print("\\nTop 10 outliers por Monetary:")
outliers.nlargest(10, "Monetary")
"""
        ),
        md(
            """\
**Interpretación de negocio:**

Estos suelen ser:
- **Mayoristas** con cantidades enormes.
- **Cuentas corporativas** con compras esporádicas pero altas.
- **Casos atípicos** que merecen atención manual del equipo de ventas.

DBSCAN te los entrega "gratis", lo cual es invaluable.

## 7. Limitaciones de DBSCAN

- En **alta dimensión** (>10), la noción de "densidad" se debilita
  (curse of dimensionality). Considera reducir con PCA antes.
- Si los clusters tienen densidades muy diferentes, un único `eps` no
  funciona bien. Considera **HDBSCAN** (versión jerárquica).
- No siempre encuentra el número de clusters que el negocio necesita.

## 8. Guardar el modelo
"""
        ),
        code(
            """\
joblib.dump(dbscan, DBSCAN_MODEL_FILE)
print(f"DBSCAN guardado en: {DBSCAN_MODEL_FILE}")
"""
        ),
        md(
            """\
## Resumen

| Aspecto | DBSCAN |
|---|---|
| Entrada principal | `eps`, `min_samples`. |
| Cómo elegir `eps` | Curva k-distance. |
| Salida especial | Etiqueta `-1` para outliers. |
| Forma de clusters | Arbitraria (no esférica). |
| Cuándo usarlo | Cuando esperas outliers genuinos o clusters no convexos. |

---

## Preguntas de Reflexión

1. ¿Por qué `min_samples` ≥ 2 × nº features es una buena heurística?
2. ¿Qué pasaría si aplicaras DBSCAN sin escalar las features?
3. Si DBSCAN encuentra 12 clusters muy pequeños, ¿qué le aconsejarías
   al negocio?
4. ¿En qué situaciones K-Means es preferible a DBSCAN, y viceversa?

> **Próximo paso:** ``08_model_comparison_and_interpretation.ipynb`` —
> comparar ambos modelos y traducir los clusters en perfiles accionables.
"""
        ),
    ]
    write_notebook("07_clustering_dbscan.ipynb", cells)


# ===========================================================================
# 08 — Comparison & Interpretation
# ===========================================================================
def build_08():
    cells = [
        md(
            """\
# 08 · Comparación e Interpretación de Segmentos

> **Objetivo:** comparar K-Means vs DBSCAN y traducir los clusters en
> perfiles accionables para el equipo de marketing.

## ¿Qué vamos a hacer?

1. Cargar ambos modelos y aplicar al mismo dataset.
2. Comparar sus métricas y distribuciones.
3. Construir el **perfil de negocio** de cada cluster.
4. Asignar **nombres** y **acciones recomendadas**.
5. Guardar el resultado final.

## ¿Cuándo es K-Means o DBSCAN la mejor elección?

| Dimensión | K-Means | DBSCAN |
|---|---|---|
| Tienes una idea clara del número de segmentos | ✅ | ❌ |
| Necesitas detectar outliers como tales | ❌ | ✅ |
| Forma de clusters | Esférica | Arbitraria |
| Sensibilidad a outliers | Alta | Robusta |
| Fácil de explicar a stakeholders | ✅ | Medio |
| Velocidad en datasets grandes | ✅ | OK |
| Dimensionalidad alta | OK | Sufre |

> **Spoiler:** en la práctica, muchas empresas usan K-Means para el
> análisis principal y DBSCAN como detector de outliers en paralelo.
"""
        ),
        code(PATH_BOOTSTRAP),
        code(
            """\
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from segmentacion.config import (
    FEATURES_DATA_FILE,
    EXTENDED_NUMERIC_FEATURES,
    KMEANS_MODEL_FILE,
    DBSCAN_MODEL_FILE,
    SEGMENTS_FILE,
    SEGMENT_PROFILES_FILE,
)
from segmentacion.features.preprocessing import build_preprocessing_pipeline
from segmentacion.models.clustering import evaluate_clustering
from segmentacion.models.profiling import build_segment_profiles, label_segments
from segmentacion.visualization.plots import plot_segment_radar
"""
        ),
        md("## 1. Cargar features y modelos"),
        code(
            """\
features = pd.read_parquet(FEATURES_DATA_FILE)
pipeline = build_preprocessing_pipeline(numeric_features=EXTENDED_NUMERIC_FEATURES, use_log=True)
X = pipeline.fit_transform(features[EXTENDED_NUMERIC_FEATURES])

kmeans = joblib.load(KMEANS_MODEL_FILE)
dbscan = joblib.load(DBSCAN_MODEL_FILE)

features["cluster_kmeans"] = kmeans.labels_
# DBSCAN no expone .predict; reusamos labels_ porque entrenamos sobre el mismo X
features["cluster_dbscan"] = dbscan.labels_
"""
        ),
        md("## 2. Comparar métricas"),
        code(
            """\
metrics_km = evaluate_clustering(X, kmeans.labels_)
metrics_db = evaluate_clustering(X, dbscan.labels_)

comparison = pd.DataFrame({"K-Means": metrics_km, "DBSCAN": metrics_db}).T
comparison.round(3)
"""
        ),
        md(
            """\
**Cómo leerla:**

- `silhouette` → más alto mejor.
- `davies_bouldin` → más bajo mejor.
- `calinski_harabasz` → más alto mejor.
- `n_noise` → solo aplica a DBSCAN.

> Frecuentemente K-Means gana en silhouette (clusters esféricos compactos
> tras el log+escalado), pero DBSCAN aporta el valor diferencial de
> aislar outliers.

## 3. Distribución de clientes por modelo
"""
        ),
        code(
            """\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
features["cluster_kmeans"].value_counts().sort_index().plot.bar(
    ax=axes[0], color="steelblue"
)
axes[0].set_title("K-Means · clientes por cluster")
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("Nº clientes")

features["cluster_dbscan"].value_counts().sort_index().plot.bar(
    ax=axes[1], color="darkorange"
)
axes[1].set_title("DBSCAN · clientes por cluster (-1 = ruido)")
axes[1].set_xlabel("Cluster")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 4. Perfilamiento del modelo principal (K-Means)

Calculamos el promedio de las features originales por cluster.
"""
        ),
        code(
            """\
profiles_km = build_segment_profiles(
    features[EXTENDED_NUMERIC_FEATURES],
    labels=features["cluster_kmeans"].values,
)
profiles_km
"""
        ),
        md(
            """\
## 5. Asignar nombres de negocio y acciones recomendadas

`label_segments` aplica una heurística basada en cuartiles relativos
de Recency, Frequency y Monetary entre los clusters.
"""
        ),
        code(
            """\
profiles_km_labeled = label_segments(profiles_km)
profiles_km_labeled[["n_customers", "share", "Recency", "Frequency", "Monetary",
                     "segment_name", "action"]]
"""
        ),
        md(
            """\
**Lectura típica de los segmentos:**

| Segmento | Característica | Acción |
|---|---|---|
| Champions | Recency baja, Frequency alta, Monetary alto | VIP, recompensas. |
| Leales | Frequency alta, Monetary medio/alto | Cross-sell, referidos. |
| Nuevos / Ocasionales | Recency baja, Frequency baja | Bienvenida, nurturing. |
| En Riesgo | Recency alta, Monetary alto histórico | Reactivación urgente. |
| Inactivos | Recency alta, Frequency y Monetary bajos | Última oferta o baja. |

## 6. Visualización tipo radar
"""
        ),
        code(
            """\
fig = plot_segment_radar(
    profiles_km_labeled,
    features=["Recency", "Frequency", "Monetary", "AvgTicket", "ProductDiversity"],
)
fig.show()
"""
        ),
        md(
            """\
> Cada polígono es un segmento. Si dos polígonos son muy similares,
> probablemente puedas fusionar esos clusters sin perder accionabilidad.

## 7. Perfilamiento de DBSCAN (incluye outliers)
"""
        ),
        code(
            """\
profiles_db = build_segment_profiles(
    features[EXTENDED_NUMERIC_FEATURES],
    labels=features["cluster_dbscan"].values,
)
profiles_db_labeled = label_segments(profiles_db)
profiles_db_labeled[["n_customers", "share", "Recency", "Frequency", "Monetary",
                     "segment_name", "action"]]
"""
        ),
        md(
            """\
> El cluster `-1` representa los **clientes atípicos** detectados por DBSCAN.
> En este dataset suelen ser mayoristas con compras grandes. Vale la pena
> manejarlos con un equipo de cuentas dedicado, no con campañas masivas.

## 8. Guardar resultados finales
"""
        ),
        code(
            """\
features.reset_index().to_parquet(SEGMENTS_FILE, index=False)

profiles_export = {
    "kmeans": profiles_km_labeled.reset_index().to_dict(orient="records"),
    "dbscan": profiles_db_labeled.reset_index().to_dict(orient="records"),
}
SEGMENT_PROFILES_FILE.write_text(json.dumps(profiles_export, indent=2, default=str))

print(f"Segmentos por cliente: {SEGMENTS_FILE}")
print(f"Perfiles JSON:        {SEGMENT_PROFILES_FILE}")
"""
        ),
        md(
            """\
## 9. Validación final con el negocio

Antes de declarar "victoria", responde:

1. ¿Cada segmento tiene un volumen mínimo accionable (≥ ~100 clientes)?
2. ¿Los nombres de los segmentos resuenan con marketing?
3. ¿Las acciones recomendadas se pueden ejecutar con las herramientas
   de email/CRM disponibles?
4. ¿Hay un plan de medición (uplift en conversión, retención, AOV)?

Si la respuesta a cualquiera es "no", itera: ajusta `k`, redefine features,
o discútelo con stakeholders.

## Resumen

- **K-Means** es la opción default, fácil de explicar.
- **DBSCAN** complementa con detección de outliers.
- Los clusters numéricos no sirven solos: hay que **traducirlos**
  a perfiles y acciones.
- La validación con el negocio es tan importante como las métricas.

---

## Preguntas de Reflexión

1. Si dos modelos dan métricas muy parecidas, ¿cómo decides cuál usar?
2. ¿Qué experimento harías para *medir* el impacto de la segmentación
   en el negocio?
3. ¿Qué riesgos éticos hay en segmentar clientes (sesgos, exclusión)?
4. Si dentro de un año el comportamiento de los clientes cambia, ¿cuándo
   y cómo re-entrenarías el modelo?

> **Cierre:** ahora abre la app de Streamlit (`app/streamlit_app.py`)
> y juega con los hiperparámetros para ver en vivo cómo cambian los
> segmentos.
"""
        ),
    ]
    write_notebook("08_model_comparison_and_interpretation.ipynb", cells)


# ---------------------------------------------------------------------------
# Ejecutar todo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generando notebooks...")
    build_01()
    build_02()
    build_03()
    build_04()
    build_05()
    build_06()
    build_07()
    build_08()
    print(f"\nListo. Notebooks creados en: {NOTEBOOKS_DIR}")

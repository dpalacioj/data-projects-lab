# Polars vs pandas — cuándo cada uno

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_segmentacion-clientes_polars.md`](../../docs/specs/SPEC_segmentacion-clientes_polars.md).

## La pregunta

**¿Qué estás optimizando cuando eliges entre Polars y pandas?** La
respuesta no es «velocidad». Es «qué pierdes si te equivocas de dialecto».

## Punto de partida: el dolor

- `pandas` sobre 1 M filas de `Online Retail II`: 4-7 s por operación,
  consumo de RAM alto, API de 15 años con convenciones que arrastran
  (`.apply`, `inplace=True`, índices).
- `Polars` sobre las mismas 1 M filas: < 1 s, lazy explícito, API
  predecible, pero las convenciones son nuevas y la comunidad de
  ejemplos es más chica.

El problema **no es cuál es más rápido**. Es que, cuando el proyecto
tiene dos sabores en el mismo pipeline, el estudiante no sabe en qué
dialecto debe estar en cada momento.

## Idea central

Regla del proyecto (sección 5.1 de la spec):

> **Polars hasta la matriz de variables; scikit-learn desde ahí.**

Traducido:

- Cargar, limpiar, construir variables → **Polars**.
- Entrenar, serializar, inferir → **pandas** (porque scikit-learn espera
  `pd.DataFrame` / `np.ndarray`) y numpy.
- La frontera es un `.to_pandas()` **explícito**, idealmente en un solo
  sitio del código.

Verificado por: `grep -r "import pandas" src/segmentacion/features/`
devuelve vacío.

## Teoría

Polars y pandas resuelven la misma clase de problema (DataFrames
tabulares) con decisiones de diseño distintas:

| Decisión | pandas | Polars |
|---|---|---|
| Índice | Primera clase (`Series.index`) | No existe; todo son columnas |
| Nulos | `NaN` para float, `NA` para object (mezcla) | `null` tipado por columna |
| Lazy | Opcional, no idiomático | `scan_parquet`, `.lazy()`, optimizador explícito |
| Expresiones | Métodos encadenados sobre `Series` | `pl.col("x")` como objeto reutilizable |
| Multithread | GIL de CPython | Rust bajo el motor |
| Fechas | `pd.Timestamp` con zonas horarias | `pl.Datetime("us")` con parseo estricto |

**McKinney (2017)**, _Python for Data Analysis_, 2.ª ed. documenta
pandas. **La documentación oficial** de Polars 2.x es la referencia
primaria (sin fuentes secundarias; muchos tutoriales viejos usan API
retirada).

## Práctica

Comparación sobre el mismo cálculo (Spearman por grupo):

```python
# pandas (lento, idiomático)
import pandas as pd
df = pd.read_parquet(ruta)
rfm = df.groupby("Customer ID").agg({"Invoice": "nunique", "Importe": "sum"})
rho = rfm.corr(method="spearman").iloc[0, 1]

# Polars (rápido, declarativo)
import polars as pl
df = pl.read_parquet(ruta)
rfm = df.group_by("Customer ID").agg(
    pl.col("Invoice").n_unique().alias("frec"),
    pl.col("Importe").sum().alias("monetario"),
)
rho = rfm.select(pl.corr("frec", "monetario", method="spearman")).item()
```

Mídelo tú:

```bash
cd projects/segmentacion-clientes
uv run python -c "
import time, polars as pl, pandas as pd
ruta = 'data/raw/online_retail_II.parquet'
t = time.time(); pd.read_parquet(ruta); print('pandas:', time.time()-t, 's')
t = time.time(); pl.read_parquet(ruta); print('polars:', time.time()-t, 's')
"
```

## Verificación

Tres pruebas que el estudiante hace solo:

1. Abrir `src/segmentacion/data/limpieza.py` y confirmar que ni una
   sola línea importa pandas.
2. Abrir `src/segmentacion/pipelines/preprocesamiento.py` y confirmar
   que recibe `pd.DataFrame` (porque es la frontera a scikit-learn).
3. En `notebooks/05_eda_automatizado.ipynb`, señalar la **única** línea
   que convierte Polars a pandas: `ventas.sample(n=100_000, seed=42).to_pandas()`.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| `df.to_pandas()` en varios sitios del paquete | La frontera se diluye y nadie sabe en qué dialecto está una función | Un solo `.to_pandas()` en el borde con sklearn |
| `df.apply(lambda r: ...)` en Polars | Cae a Python; pierdes el motor Rust | `pl.col(...).map_batches(...)` o reformula con expresiones |
| Convertir en cada notebook por pereza | Memoria duplicada; tiempo perdido | Trabaja en Polars; convierte solo si lo pide una librería externa |
| Copiar código de un tutorial de Polars 0.x | API cambió en 1.0 y en 2.0 | Documentación oficial de 2.x; nada más viejo |

## Si falla

- «Mi `pl.read_excel` tarda minutos»: ver [`troubleshooting.md`](troubleshooting.md)
  §«fastexcel vs calamine».
- «No puedo convertir una columna Polars Date a pandas Timestamp»: ver
  §«Zonas horarias y Datetime».

## Siguiente

- Próximo tema: [`esquema-al-leer.md`](esquema-al-leer.md).
- Cuaderno asociado: [`01_comprension_datos.ipynb`](../../projects/segmentacion-clientes/notebooks/01_comprension_datos.ipynb).

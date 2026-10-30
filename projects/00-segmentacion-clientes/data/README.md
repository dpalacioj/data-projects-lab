---
nav_exclude: true
---
# Datos — segmentacion-clientes

Los datos crudos **no se versionan** en este repositorio. El script
`scripts/descargar_datos.py` los descarga de UCI, verifica su integridad
y los convierte al formato de trabajo (Parquet).

## Fuente

- **Dataset:** Online Retail II.
- **Autor:** Daqing Chen (2019).
- **Repositorio:** UCI Machine Learning Repository, id **502**
  (<https://archive.ics.uci.edu/dataset/502/online+retail+ii>).
- **Licencia:** Creative Commons Attribution 4.0 International
  (CC BY 4.0 — <https://creativecommons.org/licenses/by/4.0/>).
- **Cita sugerida:** Chen, D. (2019). *Online Retail II* [Data set]. UCI
  Machine Learning Repository. <https://doi.org/10.24432/C5CG6D>.

## Qué contiene

Transacciones reales de un retailer británico de regalos entre
**2009-12-01 07:45** y **2011-12-09 12:50**: `1 067 371` filas,
`8` columnas (`Invoice`, `StockCode`, `Description`, `Quantity`,
`InvoiceDate`, `Price`, `Customer ID`, `Country`). El diccionario de
variables está en `DICCIONARIO_DATOS.md` del proyecto.

## Cómo descargarlo

Desde la raíz del proyecto:

```bash
uv sync --extra segmentacion-clientes
uv run python scripts/descargar_datos.py
```

El script:

1. Descarga `online_retail_II.zip` desde el enlace oficial de UCI.
2. Verifica el **SHA-256** del zip contra la constante `SHA256_ZIP` del
   propio script.
3. Extrae `online_retail_II.xlsx`.
4. Lee el Excel con `polars.read_excel` aplicando el esquema
   `{Invoice: Utf8, StockCode: Utf8, Customer ID: Int64}` y escribe
   `online_retail_II.parquet` con compresión zstd.

Deja los tres archivos en `data/raw/` (todos ignorados por git).

## Integridad

| Archivo | Tamaño | SHA-256 |
|---|---|---|
| `online_retail_II.zip` | 45,6 MB | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |

**Fecha de descarga de referencia:** 2026-10-16 (con la que se calcularon
las cifras de la sección 4 de `docs/specs/SPEC_segmentacion-clientes_polars.md`).

Si UCI vuelve a publicar el archivo, el hash cambiará y
`scripts/descargar_datos.py` saldrá con código `2`. En ese caso: investigar
el nuevo contenido, actualizar `SHA256_ZIP` **y** las cifras de la spec y
del material del curso al mismo tiempo. No se actualiza una sola cosa.

---
nav_exclude: true
---
# Diccionario de Datos

## Dataset Original — Online Retail II (UCI ML Repository, ID 502)

**Fuente:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
**Cobertura:** 2009-12-01 a 2011-12-09
**Granularidad:** una fila = una línea de factura.
**Total de filas:** 1 067 371 (exactas, antes de limpieza).

**Tipo declarado en la lectura:** el proyecto declara el esquema al leer
el archivo (sección 2.1 de la spec). La convención vive en
`segmentacion.data.cargar.ESQUEMA` y se aplica tanto desde Excel como
desde Parquet. Las columnas no listadas en `ESQUEMA` las infiere Polars.

| Columna | Tipo declarado en la lectura | Tipo real (post-lectura) | Descripción | Notas |
|---|---|---|---|---|
| `Invoice` | `pl.Utf8` (declarado) | `String` | Número de factura (6 dígitos). Si empieza por **`C`** indica cancelación. | Si se infiriera como entero se perderían las `C...`. Por eso se declara. |
| `StockCode` | `pl.Utf8` (declarado) | `String` | Código del producto (5 dígitos). Algunos contienen letras (ej. `POST` = correo). | 5 305 productos únicos. |
| `Description` | — (inferido) | `String` | Nombre del producto en inglés. | 4 382 filas nulas (todas con `Price = 0` y sin cliente). |
| `Quantity` | — (inferido) | `Int64` | Unidades compradas en esa línea. | Puede ser negativo (devoluciones) o nulo en errores. |
| `InvoiceDate` | — (inferido) | `Datetime(us)` | Fecha y hora de la factura. | Rango: 2009-12-01 07:45 → 2011-12-09 12:50. |
| `Price` | — (inferido) | `Float64` | Precio unitario en libras esterlinas (GBP). | 6 202 con `Price = 0`; 5 con `Price < 0`. |
| `Customer ID` | `pl.Int64` (declarado) | `Int64` (nullable) | ID anonimizado del cliente. | 243 007 filas con nulo (22,77 %). 5 943 valores distintos (incluye `null`). |
| `Country` | — (inferido) | `String` | País del cliente. | 43 países; UK domina (~91 %). |

> El archivo original viene en Excel (`.xlsx`) con dos hojas,
> `Year 2009-2010` y `Year 2010-2011`. `leer_excel_con_esquema` las
> concatena vertical. `scripts/descargar_datos.py` convierte el `.xlsx`
> a Parquet con compresión zstd en una sola pasada.

---

## Variables Derivadas (post limpieza)

### A nivel de transacción

| Columna | Tipo | Cálculo | Significado |
|---|---|---|---|
| `Importe` | `Float64` | `Quantity * Price` | Total monetario de esa línea. Lo agrega `preparar_ventas` tras aplicar las cuatro reglas de limpieza. |

### A nivel de cliente (features finales)

`construir_rfm(ventas, fecha_corte)` las devuelve en un solo
`pl.DataFrame`. Son las variables sobre las que se hace el clustering en
la Unidad 3.

| Variable | Tipo | Cálculo | Interpretación de Negocio |
|---|---|---|---|
| `recencia` | `Int64` | Días entre `fecha_corte` y `max(InvoiceDate)` del cliente. | **Bajo = activo** · alto = inactivo. |
| `antiguedad` | `Int64` | Días entre `fecha_corte` y `min(InvoiceDate)` del cliente. | Alto = cliente de larga data; bajo = reciente. |
| `frecuencia` | `UInt32` | `n_unique(Invoice)`. | **Alto = cliente fiel** · bajo = ocasional. |
| `monetario` | `Float64` | `sum(Importe)`. | **Alto = cliente valioso**. |
| `productos_distintos` | `UInt32` | `n_unique(StockCode)`. | Alto = explorador · bajo = leal a pocos productos. |

> El RFM clásico (`Recency` / `Frequency` / `Monetary`) corresponde a
> `recencia`, `frecuencia` y `monetario`. Las dos adicionales
> (`antiguedad`, `productos_distintos`) son el "RFM extendido" que U3
> usa como features.

---

## Snapshot Date

El **snapshot date** es la fecha de referencia para calcular `Recency`.

- **Convención del proyecto:** `max(InvoiceDate) + 1 día`.
- **Por qué `+1`:** evita que un cliente que compró el último día tenga
  `Recency = 0`, lo cual causa problemas con `log1p` y dificulta el
  ordenamiento.
- **Snapshot date típico para este dataset:** 2011-12-10.

---

## Notas de Calidad

| Problema | Acción tomada | Justificación |
|---|---|---|
| `CustomerID` faltante (~25%) | **Eliminar** filas. | No se pueden atribuir a un cliente; imputar sería inventar. |
| Cancelaciones (Invoice con `C`) | **Eliminar**. | No representan compras netas. Para un análisis profundo se podría restar al RFM, pero introduce complejidad. |
| `Quantity <= 0` o `Price <= 0` | **Eliminar**. | Errores o devoluciones; no aportan al perfil de cliente. |
| Duplicados exactos | **Eliminar**. | Son errores de carga, no transacciones distintas. |
| Outliers en `Quantity` y `Monetary` | **Conservar + transformar**. | Pueden ser mayoristas legítimos. La transformación `log1p` reduce su influencia sin descartarlos. |

---

## Tamaño exacto tras limpieza (contrato con la spec)

Secuencia de `preparar_ventas`:

| Etapa | Filas |
|---|---|
| Crudo | 1 067 371 |
| Tras `quitar_duplicados` | 1 033 036 |
| Tras `separar_cancelaciones` | 1 013 932 |
| Tras `exigir_cliente` | 779 495 |
| Tras `filtrar_validas` | **779 425** (73,0 %) |

- Cancelaciones (post-dedup): 19 104.
- Clientes únicos tras `exigir_cliente`: 5 942 (sin null).
- Clientes en el RFM final: 5 878 (algunos quedan sin ventas válidas tras
  el filtro de `Quantity > 0` y `Price > 0`).

> Si estas cifras no coinciden al re-ejecutar, el test
> `tests/test_cifras_referencia.py` (marcado `slow`) falla. No se ajusta
> el test; se investiga el nuevo hash en `scripts/descargar_datos.py`.

---
title: Dataset card
parent: "05 · Datos como código"
grand_parent: Tutoriales
nav_order: 2
---
# Dataset card — Online Retail II (ejemplo lleno)

> Esta dataset card documenta el caso guía del repositorio. **Al mismo tiempo
> sirve de plantilla**: así se ve una dataset card bien hecha. La versión en
> blanco para que llenes la de tu proyecto está en el
> [taller](taller.md).
>
> **Fecha de esta ficha:** 2026-10-20. Las cifras están medidas sobre el
> parquet procesado por `projects/00-segmentacion-clientes/scripts/descargar_datos.py`,
> no estimadas.

Se documenta el dato con el mismo rigor con el que se documenta el modelo. No
es burocracia: es lo que permite que otra persona —tu revisor, tú en tres
meses, un auditor— sepa qué se midió, con qué población, con qué sesgos y bajo
qué licencia.

---

## 1. Identificación

| Campo | Valor |
|---|---|
| Nombre | Online Retail II |
| Proveedor | UCI Machine Learning Repository |
| ID oficial | 502 |
| Página oficial | [archive.ics.uci.edu/dataset/502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Formato original | Excel (`.xlsx`), dos hojas (2009-2010, 2010-2011) |
| Formato que usa el curso | Parquet (una sola tabla), generado por el script de descarga |
| Licencia | **CC BY 4.0** — uso libre con atribución |
| Fecha de descarga de esta cohorte | 2026-10-15 |
| Tamaño | ~45 MB comprimido en formato Parquet |
| Dimensiones | **1 067 371 filas × 8 columnas** |

**Atribución que hay que incluir si publicas resultados:** *"Chen, D. (2019).
Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D"*

---

## 2. Qué representa cada fila

Una fila es **una línea de una factura**: una cantidad de un producto comprado
por un cliente en una fecha, en una tienda online británica de regalos entre
diciembre de 2009 y diciembre de 2011.

Un cliente puede aparecer en muchas filas (una por producto en cada factura);
una factura con 20 productos genera 20 filas.

### Columnas

| Columna | Tipo | Qué contiene | Nulos |
|---|---|---|---|
| `Invoice` | `str` | ID de factura. Si empieza por `C` es una **cancelación** (Quantity negativa). | 0 % |
| `StockCode` | `str` | ID del producto. Puede repetirse con variantes (`85123A`, `85123a`) | 0 % |
| `Description` | `str` | Descripción libre del producto en inglés. Útil para LLM como transformador. | ~0.4 % |
| `Quantity` | `int` | Unidades compradas. **Negativa en cancelaciones.** | 0 % |
| `InvoiceDate` | `datetime` | Fecha y hora de la factura. | 0 % |
| `Price` | `float` | Precio unitario, **en libras esterlinas (£)**. | 0 % |
| `Customer ID` | `float` | ID numérico del cliente. | ~22 % (compras de invitados sin login) |
| `Country` | `str` | País del cliente. 90 % Reino Unido. | 0 % |

---

## 3. Población representada y sesgos conocidos

### Qué está dentro

- **Un solo retailer** de regalos (figuritas, decoración, tazas, papelería).
- **Un solo canal:** venta online.
- **Dos años de historia** (2009-12 a 2011-12).
- **Clientes mayoritariamente del Reino Unido** (90 % de las filas).

### Qué está fuera (y lo que eso implica)

- No es representativo de retail físico, de otras categorías de producto, de
  otros países o de otros tramos temporales. Un modelo entrenado aquí **no se
  generaliza** a Amazon, a Zara o a Mercado Libre.
- Los `Customer ID` nulos son compras de "invitado" (sin login). Si tu caso de
  análisis requiere identificar clientes recurrentes, esas filas están fuera
  del alcance.
- La columna `Description` tiene errores de tipeo, abreviaciones y mezcla de
  mayúsculas/minúsculas. No está normalizada.

### Sesgos conocidos

- **Sesgo de volumen por país:** modelos que optimizan la métrica global quedarán
  dominados por el comportamiento de clientes del Reino Unido.
- **Sesgo de clientes identificados:** los que tienen `Customer ID` tienen
  comportamientos distintos (más fieles) a los invitados. Un análisis de recompra
  sobre identificados **no** se puede extrapolar a invitados.
- **Sesgo temporal:** incluye el pico de Navidad de 2010. Modelos que no
  incorporan estacionalidad explícita sobreajustan a diciembre.

---

## 4. Limitaciones y "gotchas" al cargar

| Problema | Qué significa | Cómo lo manejamos |
|---|---|---|
| Cancelaciones | Facturas con `Invoice` que empieza por `C` tienen `Quantity` negativa. **No** son errores de datos; son devoluciones reales. | Se procesan aparte en `src/segmentacion/data/limpieza.py`; no se mezclan con compras positivas sin trazabilidad. |
| `Price = 0` | Hay filas con precio cero. Pueden ser muestras gratuitas, errores de captura, o productos sin precio asignado. | Se tratan como señal de bajo valor; se mantienen en una regla con comentario. |
| `StockCode` no normalizado | Un mismo producto puede aparecer como `85123A` y `85123a`. | Antes de agregaciones, se normaliza a mayúsculas. |
| `Description` nula o críptica | Algunas filas son códigos administrativos (`AMAZON FEE`, `Adjust bad debt`). | Se filtran por reglas en `limpieza.py`; se documentan en `HALLAZGOS.md`. |
| Zona horaria de `InvoiceDate` | No está declarada. Asumimos **UTC** por defecto, pero no está confirmado por el proveedor. | Se trata como naive datetime; cualquier análisis temporal declara ese supuesto. |

---

## 5. Variables derivadas que el curso usa

Del dataset bruto, el caso guía deriva:

- `recompra` (clasificación binaria): ¿el cliente volvió a comprar en los
  próximos N días después del corte? Es la **variable objetivo** de la S03.
- **RFM** (Recencia, Frecuencia, Monto): tres variables agregadas por cliente,
  construidas a corte temporal, en `src/segmentacion/features/rfm.py`.
- Cohortes A y B: dos cortes temporales distintos para probar robustez
  temporal. Especificadas en `docs/specs/SPEC_procesamiento_U3.md`.

---

## 6. Cómo reproducir las cifras de esta ficha

```bash
# Descargar el parquet
uv run python projects/00-segmentacion-clientes/scripts/descargar_datos.py

# Las cifras citadas (1 067 371 filas, 8 columnas, nulos por columna)
uv run python -c "
import polars as pl
df = pl.read_parquet('projects/00-segmentacion-clientes/data/raw/online_retail_II.parquet')
print(df.shape)
print(df.null_count())
"
```

Si una cifra no coincide, actualiza la ficha o investiga el cambio antes de
asumir que la nueva es la buena. **La ficha es contrato tanto como el código.**

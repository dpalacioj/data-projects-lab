# Declarar el esquema al leer

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_segmentacion-clientes_polars.md`](../../docs/specs/SPEC_segmentacion-clientes_polars.md) §5.2.

## La pregunta

**¿De dónde salen los tipos de las columnas cuando el motor "los
infiere"? Y, sobre todo, ¿qué pierdes si te equivocas?**

## Punto de partida: el dolor

En Online Retail II, la columna `Invoice` contiene mayoritariamente
números de 6 dígitos (`536365`, `536366`, ...). Si Polars la lee sin
instrucciones, la castea como `Int64` y **las filas que empiezan por
`C` las convierte en null**.

Esas filas son las **cancelaciones**: 19 494 facturas que son
información útil (tasa de devolución por cliente en U3). Perderlas en
silencio es el bug más caro del flujo.

## Idea central

> _Los tipos no se infieren; se declaran. Y lo que se declara se
> comprueba al leer, no al ejecutar._

Reglas del proyecto:

1. Toda columna categórica o alfanumérica se declara como `pl.Utf8`.
2. Toda columna de identificador numérico (`Customer ID`) se declara
   explícitamente como `pl.Int64`.
3. El resto queda inferido, pero se verifica con una prueba (ver §4).

Vive en `src/segmentacion/data/cargar.py`:

```python
ESQUEMA = {"Invoice": pl.Utf8, "StockCode": pl.Utf8, "Customer ID": pl.Int64}
```

y se usa consistentemente desde Excel, desde Parquet y desde cualquier
lectura futura.

## Teoría

- **Documentación de Polars 2.x**: `read_excel` acepta
  `schema_overrides`. Es la vía oficial.
- **El Parquet mantiene tipos.** Una vez que escribes el Parquet con
  tipos correctos, la segunda ejecución ya no necesita overrides.
- **El Excel pierde tipos** entre hojas: el motor adivina por hoja.
  Un ID `C536379` en la primera hoja aparece como string; en la
  segunda, puede caer como null. El override lo fuerza en ambas.

## Práctica

En el cuaderno 01 (`01_comprension_datos.ipynb`), el primer bloque:

```python
from segmentacion.data.cargar import ESQUEMA, leer_parquet
crudo = leer_parquet(RAW_DATA_DIR / "online_retail_II.parquet")
print(crudo.schema)
```

Lo que debe salir:

```
Schema([('Invoice', String), ('StockCode', String), ('Description', String),
        ('Quantity', Int64), ('InvoiceDate', Datetime(time_unit='us', time_zone=None)),
        ('Price', Float64), ('Customer ID', Int64), ('Country', String)])
```

Si no sale, el Parquet se generó mal; se regenera con `scripts/descargar_datos.py`.

## Verificación

La prueba automática que lo garantiza:

```bash
uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_referencia.py::test_facturas_que_empiezan_por_C
```

Debe decir `== 19_494` (sin tolerancia, es un conteo). Si da 0, el
esquema no se aplicó. Si da otro número, el archivo cambió (ver §Si falla).

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| `pl.read_excel(ruta)` sin esquema | Las 19 494 cancelaciones desaparecen | `schema_overrides={"Invoice": pl.Utf8, "StockCode": pl.Utf8, "Customer ID": pl.Int64}` |
| Castear después (`.cast(pl.Utf8)`) | Ya se perdieron las filas no-numéricas | Casteo al leer, no después |
| Confiar en `df.dtypes` sin fijarlo en una prueba | Un cambio del archivo lo rompe en silencio | Prueba `slow` que verifica el schema |
| Un esquema distinto por notebook | Dos fuentes de verdad | Uno solo en `data/cargar.py::ESQUEMA` |

## Si falla

- «Mi conteo da `0` cancelaciones»: olvidaste el override. Revisa
  `cargar.py::ESQUEMA`.
- «Mi conteo da `19 500`»: incluiste las facturas que empiezan por `A`
  (6 extras); la cifra del spec es solo `C`, no `C + A`.
- «El Parquet no existe»: `uv run python scripts/descargar_datos.py`.

## Siguiente

- Próximo tema: [`04-limpieza-5-reglas.md`](04-limpieza-5-reglas.md).
- Cuaderno asociado: [`01_comprension_datos.ipynb`](../../projects/00-segmentacion-clientes/notebooks/01_comprension_datos.ipynb).

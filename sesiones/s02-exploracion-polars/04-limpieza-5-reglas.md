# Las 5 reglas de limpieza como funciones separadas

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_segmentacion-clientes_polars.md`](../../docs/specs/SPEC_segmentacion-clientes_polars.md) §5.3. Código: [`src/segmentacion/data/limpieza.py`](../../projects/00-segmentacion-clientes/src/segmentacion/data/limpieza.py).

## La pregunta

**Si una limpieza se expresa como una sola función que hace todo, ¿cómo
sabes qué regla descartó cada fila?**

## Punto de partida: el dolor

Los pipelines monolíticos (`clean_data(df)`) parecen limpios. Pero
cuando un número cambia 6 meses después, nadie recuerda qué regla era
responsable de qué filtrado. Tampoco se puede probar una regla aislada:
hay que ejecutar todo el pipeline con datos reales.

## Idea central

Una función por regla. `preparar_ventas` las encadena con `.pipe` y
registra el conteo antes/después. Lo que entra y lo que sale es
auditable con un comando.

```
quitar_duplicados      ─┐
separar_cancelaciones  ─┤
exigir_cliente         ─┤  preparar_ventas(df)  →  (ventas, cancelaciones)
filtrar_validas        ─┘
```

**Secuencia esperada** (contrato con la spec §4):

| Paso | Filas tras aplicar |
|---|---|
| Crudo | 1 067 371 |
| `quitar_duplicados` | 1 033 036 |
| `separar_cancelaciones` | 1 013 932 |
| `exigir_cliente` | 779 495 |
| `filtrar_validas` | **779 425** (73,0 %) |

Y en paralelo: 19 104 cancelaciones a `data/processed/cancelaciones.parquet`
(insumo de U3).

## Teoría

Las 5 funciones aceptan tanto `pl.DataFrame` como `pl.LazyFrame`
(`Frame = pl.DataFrame | pl.LazyFrame`). La razón es la lección de la
spec: _«cuadernos eager, scripts lazy»_.

- **Eager** (`pl.DataFrame`): el estudiante ve las filas tras cada paso.
- **Lazy** (`pl.LazyFrame`): el optimizador reordena, fusiona y
  minimiza memoria.

Las dos ejecutan el mismo código.

### Separar cancelaciones **antes** de exigir cliente

Decisión que a veces se discute: separar las cancelaciones **antes** de
exigir `Customer ID`. Esto deja algunas cancelaciones sin cliente
(irrelevantes para RFM) en el parquet de cancelaciones, pero mantiene
simétrico el flujo: la factura `C` se identifica por su prefijo, no por
tener cliente.

Si se invirtiera el orden, cancelaciones sin cliente se perderían en
`exigir_cliente`, y U3 no podría calcular `tasa_devolucion` para los
clientes afectados (sesgo).

## Práctica

```python
from segmentacion.data.cargar import leer_parquet
from segmentacion.data.limpieza import preparar_ventas

ventas, cancelaciones = preparar_ventas(leer_parquet(parquet))
# Log del logger:
# [HH:MM:SS] INFO  segmentacion.data.limpieza | pipeline de limpieza:
#   1 067 371 -> sin duplicados 1 033 036 -> sin cancelaciones 1 013 932
#   -> con cliente 779 495 -> validas 779 425 (73,0% del original)
```

En el cuaderno 02 (`02_limpieza.ipynb`) se escribe el resultado:

```python
ventas.write_parquet("data/processed/ventas_validas.parquet")
cancelaciones.write_parquet("data/processed/cancelaciones.parquet")
```

## Verificación

Dos pruebas, dos velocidades:

1. **Rápida** (sin disco, 6 filas inventadas):
   ```bash
   uv run pytest projects/00-segmentacion-clientes/tests/test_limpieza.py
   ```
   Debe pasar en < 1 s.

2. **Lenta** (sobre el parquet real, 20 asserts):
   ```bash
   uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_referencia.py
   ```
   Reproduce las cifras del §4 de la spec: conteos con tolerancia 0,
   correlaciones con tolerancia 1e-3, media de `Price` con 5e-3
   (half-LSD de la precisión del OVA).

## HALLAZGOS.md como contrato con U3

El cuaderno 04 (EDA) deja [`HALLAZGOS.md`](../../projects/00-segmentacion-clientes/HALLAZGOS.md)
con 7 filas: hallazgo + evidencia + decisión. Cada fila es un contrato
con la Unidad 3:

| Hallazgo | Decisión | Dónde se aplica en U3 |
|---|---|---|
| 243 007 sin cliente, todas con `Price = 0` | `exigir_cliente` | `data/limpieza.py` (ya) |
| 19 494 cancelaciones | separar | `features/devoluciones.py` (U3) |
| `Price` con cola larga | `log1p` | `pipelines/preprocesamiento.py::NUM_LOG` |
| `Quantity` con cola larga | `log1p` | ídem |
| Pearson ≈ 0, Spearman −0,43 | preferir ranks | sí, con `log1p` + `StandardScaler` |
| 20 % superior = 77,2 % | fichas diferenciadas | `models/profiling.py` |
| Sábado inexistente | no modelar día | referencia |

**La fila 2 (cancelaciones) es la única que no se materializa en U2.**
Vive en el parquet `cancelaciones.parquet` y espera a U3.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| Una sola función `clean(df)` monolítica | No se audita cada regla | 5 funciones + `preparar_ventas` que las encadena |
| `df = df[df.Quantity > 0]` sin log | Perdemos la cuenta | `.filter(...)` + log del conteo |
| Reescribir las reglas en `scripts/figuras_u2.py` | Dos fuentes de verdad | El script importa del paquete |
| Eliminar las cancelaciones | U3 las necesita | `separar_cancelaciones` las guarda aparte |

## Si falla

- «Mi secuencia no da 779 425»: ver [`troubleshooting.md`](troubleshooting.md)
  §«Las cifras no coinciden con la spec».
- «`preparar_ventas` no loggea»: el `uv run` necesita `--extra dev`
  para pytest, pero el logger funciona sin dev. Si no sale, revisa
  nivel del logger en `utils/logger.py`.

## Siguiente

- Próximo tema: [`eda-manual-vs-auto.md`](eda-manual-vs-auto.md).
- Cuaderno asociado: [`02_limpieza.ipynb`](../../projects/00-segmentacion-clientes/notebooks/02_limpieza.ipynb).

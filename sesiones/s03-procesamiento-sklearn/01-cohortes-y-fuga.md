---
title: "01 · Cohortes y fuga"
parent: "S03 · Procesamiento, pipelines, modelos"
grand_parent: Sesiones
nav_order: 1
---
# Cohortes temporales y las dos fugas

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_procesamiento_U3.md`](../../docs/specs/SPEC_procesamiento_U3.md) §4 y §9. Cuaderno: [`06_particion_y_fuga.ipynb`](../../projects/00-segmentacion-clientes/notebooks/06_particion_y_fuga.ipynb). Código: [`src/segmentacion/features/cohortes.py`](../../projects/00-segmentacion-clientes/src/segmentacion/features/cohortes.py).

## La pregunta

**¿Cómo construyes un conjunto de entrenamiento y uno de prueba cuando
los datos son temporales y cualquier información del futuro contamina
el ejercicio?**

## Punto de partida: el dolor

Un pipeline sin disciplina temporal:

```python
X = df.drop(columns=["recompra"])
y = df["recompra"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Esto funciona para dígitos MNIST. Para recompra, es un error. El
modelo ve clientes que compraron en septiembre al entrenarse para
predecir qué pasa en septiembre. El AUC de validación es alto; el
AUC en producción, mediocre.

**Para datos con eje temporal, la partición es por fecha.**

## Idea central

Dos **cohortes** disjuntas por un corte temporal:

- **Cohorte A** (corte **2011-06-01**): entrenamiento + validación cruzada.
- **Cohorte B** (corte **2011-09-01**): prueba.

Cada cohorte contiene **un cliente por fila**, con variables calculadas
**solo con compras anteriores al corte** y etiqueta de **recompra en
los 90 días posteriores al corte**.

Firma pública:

```python
cohorte(ventas, cancelaciones, corte, horizonte=90) -> pl.DataFrame
```

Y la línea final, obligatoria:

```python
return features.sort("Customer ID")
```

## Teoría

### Horizonte abierto por la derecha

La etiqueta de recompra se evalúa en `[corte, corte + 90 días)`. Es
**abierto por la derecha**: una compra el día `corte + 90` no cuenta.
Es una convención — pero una convención fijada en el código, no un
accidente.

### Variables incluidas en la cohorte

| Columna | Qué mide | Cálculo |
|---|---|---|
| `recencia` | Días desde la última compra pre-corte | `(corte − max(InvoiceDate < corte)).days` |
| `antiguedad` | Días desde la primera compra pre-corte | `(corte − min(InvoiceDate < corte)).days` |
| `frecuencia` | Número de facturas únicas pre-corte | `n_unique(Invoice)` |
| `monetario` | Suma de `Importe` pre-corte | `sum(Importe)` |
| `productos_distintos` | `StockCode` únicos pre-corte | `n_unique(StockCode)` |
| `pais` | `UK` o `otro` | del primer `Country` pre-corte |
| `tasa_devolucion` | `n_cancelaciones_pre / frecuencia` | `features.devoluciones.tasa_devolucion(...)` |
| `recompra` | 0 o 1 | `features.etiquetas.etiqueta_recompra(...)` |
| `recencia_fuga` | Días desde la última compra **sin filtrar** | diagnóstico; **no** se usa en modelos |

### Reproducibilidad: por qué el `.sort`

`polars.unique()` **no conserva el orden**. Sin un sort explícito al
final, el dataframe tiene filas en orden distinto entre ejecuciones.
Esto afecta a:

- Los splits de `StratifiedKFold` (que dependen del orden al barajar).
- Los árboles del bosque (que dividen según el orden en que ven las
  filas al construir cada split).

Con `.sort("Customer ID")` + `random_state=SEMILLA`, dos ejecuciones
producen el mismo bosque.

> Caveat observado: con `n_jobs=-1` el bosque sigue teniendo
> fluctuaciones a nivel ~2e-16 por el orden de agregación de hilos.
> La prueba `test_reproducibilidad.py` usa `n_jobs=1` para alcanzar
> igualdad exacta. Las predicciones **lógicas** son idénticas; las
> **numéricas** empatan a 10 decimales.

## Práctica

```python
from datetime import datetime
from segmentacion.data.cargar import leer_parquet
from segmentacion.data.limpieza import preparar_ventas
from segmentacion.features.cohortes import cohorte

ventas, canc = preparar_ventas(leer_parquet(parquet), log=False)

A = cohorte(ventas, canc, datetime(2011, 6, 1), horizonte=90)
B = cohorte(ventas, canc, datetime(2011, 9, 1), horizonte=90)

# Cifras del §9:
# A: 4 933 clientes | 1 621 positivos | 32,9 %
# B: 5 249 clientes | 2 241 positivos | 42,7 %
```

## Verificación

```bash
# Rápido (sin disco)
uv run pytest projects/00-segmentacion-clientes/tests/features/test_cohortes.py
uv run pytest projects/00-segmentacion-clientes/tests/features/test_devoluciones.py
uv run pytest projects/00-segmentacion-clientes/tests/features/test_etiquetas.py

# Lento (contra el parquet real; reproduce §9)
uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_recompra.py::test_cohorte_A_tamanio
uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_recompra.py::test_cohorte_B_tamanio
```

## Fuga 1 — Variable con información del futuro

`cohorte` incluye deliberadamente `recencia_fuga` como variable de
**demostración**: se calcula sobre toda la serie de ventas (sin
filtrar), así que para clientes que compraron después del corte, su
valor es negativo.

Entrenar la regresión logística con `recencia_fuga` en vez de `recencia`:

| Modelo | AUC CV (A) | AUC (B) |
|---|---|---|
| Honesto (`recencia` filtrada) | 0,817 | **0,795** |
| Con fuga (`recencia_fuga`) | 0,881 | **0,955** |

Un salto de 0,16 en AUC sin cambiar la arquitectura es casi siempre
fuga. Nunca es «el modelo mejoró». La demostración vive en
`evaluation/metricas.fuga_recencia(...)` y la prueba
`test_fuga_recencia_infla_auc`.

## Fuga 2 — Preprocesamiento sobre A ∪ B

El `StandardScaler` ajustado sobre A ∪ B aprende la media global:

| Qué ajusta el scaler | Media de `recencia` que aprende |
|---|---|
| Solo A (correcto) | **168,7** días |
| A ∪ B (fuga) | **186,9** días |

La diferencia es sutil pero contamina la calibración. Se demuestra con
`evaluation/metricas.fuga_escalado(...)` y la prueba
`test_fuga_escalado_medias`.

El contratacanazo técnico: `Pipeline(preprocesador, modelo)` + `fit(X_A)`
garantiza por construcción que el scaler nunca ve B.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| `train_test_split(..., random_state=42)` sobre datos temporales | El modelo ve compras posteriores a la fecha de inferencia | Cohortes por corte temporal |
| Incluir `recencia_fuga` en los features del modelo | AUC infla, el modelo es inútil | Solo para la demostración del cuaderno 06 |
| Reordenar la cohorte después del `.sort` | Reproducibilidad perdida sin aviso | El sort es la última línea; nada después |
| «Corte» como un `datetime` sin huso | Nulls silenciosos en comparaciones | `datetime(2011, 6, 1)` naive, consistente con `InvoiceDate` |

## Si falla

- «Mis conteos no coinciden con 4 933 / 5 249»: ver [`07-troubleshooting.md`](07-troubleshooting.md)
  §«Cifras del §9 no coinciden».
- «Me da fuga sin usar `recencia_fuga`»: alguien debe estar usando
  `train_test_split` dentro del pipeline; busca en el cuaderno.

## Siguiente

- Próximo tema: [`02-pipelines-sin-fuga.md`](02-pipelines-sin-fuga.md).
- Cuaderno asociado: [`06_particion_y_fuga.ipynb`](../../projects/00-segmentacion-clientes/notebooks/06_particion_y_fuga.ipynb).

---
title: "05 · EDA manual vs. automatizado"
parent: "S02 · Exploración con Polars"
grand_parent: Sesiones
nav_order: 5
---
# EDA manual vs. automatizado

> Fecha de referencia: **2026-10-19**. Spec: [`SPEC_segmentacion-clientes_polars.md`](../../docs/specs/SPEC_segmentacion-clientes_polars.md) §4 (cifras). Cuadernos 04 y 05 del proyecto.

## La pregunta

**Si un informe HTML autogenerado me dice todo, ¿para qué sirve el EDA
manual?** Y, al revés, **¿cuándo el EDA automatizado me engaña?**

## Punto de partida: el dolor

Dos reflejos opuestos:

- _«Dale a `fg-data-profiling`, lee el HTML, y listo.»_ Resultado: el
  estudiante acepta correlaciones espurias, no cuestiona los nulos,
  no distingue cola larga de cola corta.
- _«Yo lo hago todo a mano, controlo cada número.»_ Resultado: 4 horas
  después tienes 3 gráficas, 90 % del tiempo en formato.

Ninguno enseña. El manual te obliga a decidir qué ver; el automatizado
te da un segundo par de ojos.

## Idea central

- **Manual** cuando el objetivo es decidir una regla de limpieza o una
  transformación (cuartiles, Tukey, Pearson vs. Spearman).
- **Automatizado** cuando el objetivo es compartir con un stakeholder
  o cuando quieres comparar «lo que esperabas» con «lo que un informe
  automático dice».

Y una regla práctica: **el EDA automático nunca va contra 1 M filas**.
Siempre contra una muestra reproducible (`sample(n=100_000, seed=42)`).

## Teoría

**EDA manual** cubre 4 bloques en el cuaderno 04:

1. **Univariado**: cuartiles + cercas de Tukey.
   - `Price`: Q3 = 3,75; cerca superior = 7,50; **8,40 %** de las filas
     la supera.
   - `Quantity`: cerca superior = 27; **6,56 %** la supera.
2. **Temporal**: ventas por mes + mapa día×hora.
   - 400 filas en sábado en 2 años → ignorar como clase propia.
3. **Asociaciones**: Pearson vs. Spearman sobre 779 425 filas.
   - Pearson = **−0,005**; Spearman = **−0,428**.
   - La lección: Spearman mide orden, Pearson mide valor. Cuando los
     datos tienen cola larga, el valor del Pearson es ruido; el ranking
     dice la verdad.
4. **Nivel cliente**:
   - Spearman frecuencia–monetario = **0,858** (altísima).
   - 20 % superior de clientes = **77,2 %** del total.

**EDA automatizado**:

- **`skimpy`** sobre Polars (sin convertir): resumen rápido, tipos,
  distribuciones, nulos. Es `summary()` de R, con decoración.
- **`fg-data-profiling`** (fork de `ydata-profiling` 4.20; se importa
  como `data_profiling`): HTML completo con interacciones, matriz de
  correlaciones, muestra de valores extremos.

## Práctica

### Manual (en el cuaderno 04)

```python
from segmentacion.eda.resumen import cuartiles, cercas_tukey, asociaciones

print(cuartiles(ventas, "Price"))
print(cercas_tukey(ventas, "Price"))     # (inf, sup, fraccion_fuera)
print(asociaciones(ventas, "Quantity", "Price"))  # {'pearson': ..., 'spearman': ...}
```

### Automatizado (en el cuaderno 05)

```python
from skimpy import skim
skim(ventas)                                       # directo sobre Polars

muestra = ventas.sample(n=100_000, seed=42).to_pandas()  # frontera única a pandas
from data_profiling import ProfileReport
ProfileReport(muestra, minimal=True).to_file(REPORTS / "perfil_muestra.html")
```

El HTML queda gitignored (2,2 MB) para no inflar el repo.

## Verificación

```bash
# Las 32 cifras del §4 reproducidas sobre el parquet real
uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_referencia.py

# Las 4 figuras de la spec con la paleta correcta
uv run python projects/00-segmentacion-clientes/scripts/figuras_u2.py \
    projects/00-segmentacion-clientes/data/raw/online_retail_II.parquet \
    projects/00-segmentacion-clientes/reports/figures
```

Debe imprimir:
`cerca Tukey Price=7.50 (8.40%); pearson=-0.005; spearman=-0.428; sabado=400`.

## Pearson vs. Spearman: el ejemplo del curso

- Pearson mide **asociación lineal** sobre valores crudos.
- Spearman mide **asociación monótona** sobre rangos.

Cuando la distribución tiene cola larga (nuestro caso: `Price` llega a
10 953,50 £), un puñado de valores extremos domina Pearson y lo lleva
a cero por compensación de signos. Spearman, al operar sobre rangos,
no se deja llevar.

**La lección es doble**: una, la elección estadística; dos, por qué U3
pide `log1p` antes de escalar (comprime la cola; estabiliza ambas
correlaciones).

## RFM extendido como puente a U3

En el cuaderno 03 se construye el RFM extendido con fecha de corte:

```python
from segmentacion.features.rfm import construir_rfm
rfm = construir_rfm(ventas, fecha_snapshot)  # columnas:
#   Customer ID, recencia, antiguedad, frecuencia, monetario, productos_distintos
```

**Importante**: `construir_rfm` **no** devuelve `recencia_fuga`
(variable que incluye datos posteriores al corte). Esa variable solo
existe en `features/cohortes.cohorte` de U3, para la demostración de
fuga del cuaderno 06.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| Correr `fg-data-profiling` sobre 1 M filas | 20 min + HTML de 50 MB | `sample(n=100_000, seed=42)` |
| Fijar la semilla a `None` | Perfiles distintos entre ejecuciones | `seed=42` siempre |
| Usar Pearson para datos sesgados | Decidirás con un número que es ruido | Pearson **y** Spearman; discute por qué difieren |
| Guardar el HTML en Git | 2 MB por corrida | `.gitignore` lo cubre; se regenera |

## Si falla

- «Mi perfil HTML da `ModuleNotFoundError: fg_data_profiling`»: el
  paquete se importa como `data_profiling`. Ver [`07-troubleshooting.md`](07-troubleshooting.md).
- «La correlación Spearman no coincide con el spec»: verifica que
  trabajas sobre las 779 425 filas válidas (post-`preparar_ventas`),
  no sobre el crudo.

## Siguiente

- Taller: [`06-taller.md`](06-taller.md).
- Cuadernos asociados: [`04_eda.ipynb`](../../projects/00-segmentacion-clientes/notebooks/04_eda.ipynb),
  [`03_variables_cliente.ipynb`](../../projects/00-segmentacion-clientes/notebooks/03_variables_cliente.ipynb),
  [`05_eda_automatizado.ipynb`](../../projects/00-segmentacion-clientes/notebooks/05_eda_automatizado.ipynb).

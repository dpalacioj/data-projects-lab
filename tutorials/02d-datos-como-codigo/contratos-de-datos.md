# Contratos de datos: los tres niveles

> **Pandera** es la librería que usamos aquí para expresar contratos. Instala con
> `uv pip install pandera` si todavía no lo tienes.

Un contrato de datos es un conjunto de reglas que el `pipeline` verifica **antes
de entrenar o predecir**. Si una regla falla, el `pipeline` se detiene con un
error claro, en vez de producir un número confiable sobre un dato que ya no es
el mismo.

La separación de reglas en tres niveles es lo que distingue un contrato que
atrapa incidentes reales de uno que es decorativo.

---

## Nivel 1 — Por fila (`row-level`)

Cada fila se evalúa de forma independiente. Son las reglas más fáciles de
escribir y las que casi todos los tutoriales enseñan.

**Ejemplos:**
- `Price > 0`
- `Quantity ∈ ℤ⁺`
- `CustomerID` no es nulo
- `Category` ∈ {A, B, C, D}
- `email` matches regex

**En Pandera:**

```python
import pandera.pandas as pa

esquema = pa.DataFrameSchema({
    "Price":      pa.Column(float, checks=pa.Check.gt(0)),
    "Quantity":   pa.Column(int,   checks=pa.Check.gt(0)),
    "CustomerID": pa.Column(int,   nullable=False),
    "Category":   pa.Column(str,   checks=pa.Check.isin(["A", "B", "C", "D"])),
})
```

**Qué atrapan:** errores de captura, valores imposibles, formatos corruptos.
**Qué NO atrapan:** cambios sistemáticos (todos los precios se multiplicaron por
100, pero cada precio individual sigue siendo positivo), cambios de proporción,
drift de distribución.

---

## Nivel 2 — Por distribución (`distribution-level`)

Reglas que evalúan **el lote completo**, no fila por fila. Son las que menos
gente escribe y las que atrapan los incidentes más caros.

**Ejemplos:**
- La fracción de filas con `Price > 1000` no debe superar el 1 %.
- Al menos 10 000 filas en cada lote diario (control de volumen).
- La cardinalidad de `Country` ∈ [1, 50].
- La fracción de nulos en `CustomerID` no supera el 5 %.
- La media de `Price` cae en el rango histórico [𝜇 ± 3σ].

**En Pandera:**

```python
esquema = pa.DataFrameSchema({
    "Price": pa.Column(float, checks=[
        pa.Check.gt(0),                                    # nivel 1
        pa.Check(lambda s: (s > 1000).mean() < 0.01,       # nivel 2
                 error="más del 1% de precios > 1000: revisa unidades"),
    ]),
}, checks=[
    pa.Check(lambda df: len(df) >= 10_000,                 # nivel 2
             error="lote con menos de 10k filas"),
])
```

**Qué atrapan:** cambios de unidades, cambios de escala, degradación del
proveedor (de repente el 30 % de los `CustomerID` llegan nulos), reducciones
súbitas de volumen.

**El nivel 2 es la línea defensiva real.** Un cambio sistemático de unidades
(dólares → centavos) pasa cada regla fila-a-fila. Pero la fracción de precios
"anormalmente altos" se dispara y la detectas al instante.

---

## Nivel 3 — Entre columnas (`cross-column`)

Reglas que involucran **dos o más columnas** de la misma fila o del mismo lote.

**Ejemplos:**
- `fecha_fin >= fecha_inicio` (ordenamiento temporal).
- `Revenue == Price * Quantity` (integridad aritmética).
- Si `Category == "digital"`, entonces `shipping_cost == 0`.
- La correlación entre `Price` y `Quantity` no cambia bruscamente de signo.

**En Pandera:**

```python
esquema = pa.DataFrameSchema({
    "Price":    pa.Column(float, checks=pa.Check.gt(0)),
    "Quantity": pa.Column(int,   checks=pa.Check.gt(0)),
    "Revenue":  pa.Column(float),
}, checks=[
    pa.Check(
        lambda df: ((df["Revenue"] - df["Price"] * df["Quantity"]).abs() < 0.01).all(),
        error="Revenue no cuadra con Price * Quantity"
    ),
])
```

**Qué atrapan:** inconsistencias internas de un registro, errores de ETL que
rompen la relación entre columnas, cambios de significado en una columna
dependiente de otra.

---

## Cómo ordenar reglas en un contrato

- **Siempre al menos una regla de cada nivel.** Si tu contrato tiene solo nivel
  1, no te protege de lo importante.
- **Cada regla con un comentario de una línea que diga qué protege.** Una regla
  sin justificación es una que alguien relajará en seis meses para desbloquear
  un pipeline.
- **El contrato vive junto al código**, en un archivo versionado (p. ej.
  `src/<proyecto>/data/contrato.py`). No es un notebook ni un documento Word.
- **Hay tests que verifican el contrato con fixtures rotos.** Para cada regla,
  escribe un pequeño DataFrame que la rompa y confirma que el contrato lo
  rechaza. Es el "control negativo": si tu contrato no rechaza un fixture roto,
  está apagado.

---

## Límite honesto de los contratos

Un contrato estático **no reemplaza al monitoreo**. Puede atrapar incidentes
groseros, pero cambios sutiles (una tienda nueva que empieza a enviar precios
levemente sesgados) requieren instrumentación distinta: comparar la
distribución del lote actual contra una histórica de referencia (data drift).
Eso se ve en el módulo siguiente.

Para este curso: **con el nivel 2 bien escrito ya atrapas el 80 % de los
desastres**.

---

## Para profundizar

- Documentación de Pandera: https://pandera.readthedocs.io/
- Alternativa activa en 2026: Great Expectations Core 1.x (más pesada, con UI).
- MLOps-Course sesión 2 (el original de donde adaptamos este tutorial): https://github.com/dpalacioj/MLOps-Course/tree/main/sesiones/s02-datos

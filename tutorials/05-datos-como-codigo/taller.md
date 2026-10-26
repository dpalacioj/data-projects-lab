---
title: Taller
parent: "05 · Datos como código"
grand_parent: Tutoriales
nav_order: 3
---
# Taller 02d — Dataset card + contrato de datos de tu proyecto

> Se puede terminar en la clase o quedar como tarea. **Entregable:** dos
> archivos commiteados a tu repositorio de proyecto: `dataset-card.md` y
> `contrato.py` (o `.ipynb` si todavía no tienes paquete).

---

## 1. Dataset card de tu proyecto (30 min)

Toma el dataset con el que estás trabajando en tu proyecto (si todavía no lo
elegiste, puedes usar el de iris o de titanic solo para esta primera iteración).

Usa [`dataset-card.md`](dataset-card.md) de este tutorial como plantilla rellena
y escribe una **con tus datos**. Debe incluir, como mínimo:

- [ ] **1. Identificación** — nombre, proveedor, URL oficial, licencia, fecha de
  descarga, dimensiones, formato.
- [ ] **2. Qué representa cada fila** y **qué es cada columna** (tipo, qué
  contiene, % de nulos).
- [ ] **3. Población representada y sesgos conocidos** — qué está dentro, qué
  está fuera, al menos dos sesgos específicos que un modelo entrenado aquí no
  podrá corregir.
- [ ] **4. Limitaciones y gotchas** al cargar — al menos tres cosas con las que
  tropezaste y cómo las manejas.

**Qué NO cuenta:** una tabla de 10 líneas sin pensar. Si tus "sesgos conocidos"
dicen "ninguno", no leíste bien el dato.

---

## 2. Contrato con 3 reglas, una por nivel (30 min)

Escribe un contrato de Pandera para tu dataset. Mínimo **tres reglas**, una por
cada nivel:

### Regla 1 — nivel por fila

Una regla que valide algo de **cada fila individual**. Ejemplos:

- Un precio es positivo.
- Un ID categórico está en un conjunto cerrado.
- Un campo obligatorio no es nulo.

### Regla 2 — nivel por distribución

Una regla que mire el **lote completo**. Ejemplos:

- La fracción de nulos en una columna no supera un umbral.
- El tamaño del lote está en un rango esperado.
- La media de una variable continua cae en un rango histórico.

### Regla 3 — nivel entre columnas

Una regla que involucre **dos o más columnas** de la misma fila. Ejemplos:

- `fecha_fin >= fecha_inicio`.
- `total == subtotal + impuesto`.
- Si una categórica es X, otra columna debe cumplir Y.

**Cada regla con un comentario de una línea que diga qué protege.** Una regla
sin justificación es una que alguien relajará en seis meses.

### Plantilla

```python
# contrato.py
import pandera.pandas as pa

esquema = pa.DataFrameSchema(
    columns={
        "<col1>": pa.Column(<tipo>, checks=[
            # Regla 1 (por fila): protege contra <qué>
            pa.Check.<...>(...),
        ]),
        # ...
    },
    checks=[
        # Regla 2 (por distribución): protege contra <qué>
        pa.Check(lambda df: <...>, error="<mensaje accionable>"),
        # Regla 3 (entre columnas): protege contra <qué>
        pa.Check(lambda df: <...>, error="<mensaje accionable>"),
    ],
)


def validar(df):
    """Levanta SchemaError si el DataFrame no respeta el contrato."""
    return esquema.validate(df)
```

---

## 3. Verifica que tu contrato funciona (contra fixtures rotos)

Para cada regla, construye un pequeño `DataFrame` que la rompa y confirma que
el contrato lo rechaza. **Es el control negativo.** Si tu contrato no rechaza
un fixture roto, está apagado.

```python
# en una celda del notebook o en un test
import pandas as pd
from contrato import validar

# Caso que rompe la regla 1 (por fila)
df_malo_1 = pd.DataFrame({"precio": [-5], ...})
try:
    validar(df_malo_1)
    print("FALLO: el contrato NO atrapo el error de nivel 1")
except Exception as e:
    print(f"OK: regla 1 funciona ({type(e).__name__})")

# Repite para reglas 2 y 3
```

---

## 4. Criterios de aceptación

- [ ] `dataset-card.md` tiene las 4 secciones mínimas.
- [ ] `contrato.py` tiene 3 reglas, una por nivel, cada una con comentario.
- [ ] Hay evidencia de que cada regla rechaza un fixture roto construido para
  romperla específicamente.

Si llegas hasta aquí, tu proyecto ya resiste la clase de incidente más común en
producción: alguien cambia algo arriba del `pipeline` y nadie se da cuenta.

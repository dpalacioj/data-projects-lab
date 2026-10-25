# Tutorial 04: Limpieza de datos sucios en Polars

> Patrón del repositorio: **ejercicio + solución separados**. Imita la
> estructura de `projects/d1-ejercicio-preprocesamiento/` sobre un dataset
> sintético inventado específicamente para este taller (nada se toma
> de datasets externos).

## Objetivo pedagógico

Que el estudiante aplique las reglas de limpieza más comunes en un
dataset con los problemas más frecuentes del mundo real, usando
**Polars 2.x** desde el principio. Los problemas son los que la Unidad
2 del módulo encontró en Online Retail II, replicados aquí a escala
pequeña (200 filas) para que la ejecución sea instantánea y se pueda
inspeccionar cada paso.

## Lo que vas a aprender

- Normalizar texto con mayúsculas/minúsculas inconsistentes.
- Parsear fechas con **dos formatos distintos** en la misma columna.
- Detectar **valores centinela** (`-999`, `"NA"`, `"unknown"`).
- Separar precios con dos monedas marcadas en la misma celda.
- Decidir qué duplicados son errores de carga y cuáles no.
- Construir una pequeña **tabla de hallazgos** antes de modificar nada.

## Archivos del tutorial

- `data_sinteticos.py` — genera `datos_sucios.csv` con `seed=42` (idempotente).
- `01_ejercicio.ipynb` — enunciado con celdas `TODO` para el estudiante.
- `02_solucion.ipynb` — solución comentada en Polars 2.x.

## Cómo ejecutarlo

```bash
cd tutorials/04-limpieza-datos-sucios-polars
uv run python data_sinteticos.py          # genera datos_sucios.csv
uv run jupyter lab 01_ejercicio.ipynb      # resolver el ejercicio
uv run jupyter lab 02_solucion.ipynb       # consultar la solución
```

El CSV sintético es pequeño (200 filas, ~15 KB) y se puede commitear.

## Para después

Una vez cómodo con este tutorial, el flujo real sobre Online Retail II
está en `projects/00-segmentacion-clientes/notebooks/02_limpieza.ipynb`
(mismo patrón, pero sobre el dataset de la Unidad 2 del módulo).

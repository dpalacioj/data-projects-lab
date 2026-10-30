---
title: "S01 · Encuadre CRISP-DM"
parent: Sesiones
nav_order: 1
---
# S01 · Encuadre del curso: CRISP-DM como columna vertebral

Este archivo **no** es un guión de 4 horas. Es el **mapa** que un
estudiante lee en la primera semana para entender qué recorre el
módulo y por qué CRISP-DM aparece una y otra vez.

El contenido profundo de cada fase vive en las sesiones y proyectos
correspondientes (ver tabla). Esta página es la aguja que apunta.

---

## ¿Qué es CRISP-DM?

**CRISP-DM** (*Cross-Industry Standard Process for Data Mining*, Chapman
et al., 2000) es un marco de referencia que describe un proyecto de
ciencia de datos como **seis fases iterativas**, no como una línea
recta:

1. **Comprensión del negocio** — Formular el objetivo del análisis.
2. **Comprensión de los datos** — Diagnosticar qué hay, qué falta, qué miente.
3. **Preparación de los datos** — Limpiar, construir variables, dejar la tabla modelable.
4. **Modelado** — Entrenar, comparar, ajustar.
5. **Evaluación** — ¿Resolvimos el problema de negocio, no solo el técnico?
6. **Despliegue** — Entregar algo que alguien pueda usar.

Es **cíclico**: cuando la fase 3 descubre un supuesto roto, se vuelve a
la 1. Esto es sano, no un fracaso.

---

## Por qué este curso lo usa como columna vertebral

Porque **la mayoría de los proyectos de ciencia de datos no fracasan por
el modelo** — fracasan en las fases 1 y 2 (objetivo mal definido, datos
mal entendidos) y en la 6 (nadie recoge el entregable). Un curso que
solo cubriera "modelado" dejaría fuera exactamente el 80% del trabajo
real.

Este módulo recorre explícitamente las fases 1 a 5 sobre un caso real
(Online Retail II), y practica la fase 6 (entregable) sobre un proyecto
distinto (CNN sobre Fashion-MNIST).

---

## Mapa: dónde se practica cada fase

| Fase CRISP-DM | Dónde se trabaja | Material |
|---|---|---|
| **1. Negocio** | Discusión guiada en S02 + taller individual | [`s02-exploracion-polars/01-crisp-dm-fase1.md`](../s02-exploracion-polars/01-crisp-dm-fase1.md) · [`segmentacion-clientes/notebooks/00_objetivo_de_analisis.ipynb`](../../projects/00-segmentacion-clientes/notebooks/00_objetivo_de_analisis.ipynb) |
| **2. Datos** | S02 completa | [`s02-exploracion-polars/`](../s02-exploracion-polars/) · notebooks 01-05 del caso guía |
| **3. Preparación** | S03 (parte 1) | [`s03-procesamiento-sklearn/02-pipelines-sin-fuga.md`](../s03-procesamiento-sklearn/02-pipelines-sin-fuga.md) · notebooks 06-07 del caso guía |
| **4. Modelado** | S03 (parte 2) | [`s03-procesamiento-sklearn/03-modelos-y-umbral.md`](../s03-procesamiento-sklearn/03-modelos-y-umbral.md) · notebooks 08-10 del caso guía |
| **5. Evaluación** | S03 (cierre) + taller | [`s03-procesamiento-sklearn/06-taller.md`](../s03-procesamiento-sklearn/06-taller.md) |
| **6. Entrega** | Proyecto CNN (notebook → modular → Streamlit → FastAPI) | [`projects/04-reconocimiento-imagenes/`](../../projects/04-reconocimiento-imagenes/) |

> **Nota.** No hay sesión aparte "S01 de 4 horas sobre CRISP-DM". La
> fase 1 se trabaja al inicio de S02 porque CRISP-DM solo cobra sentido
> **sobre un caso concreto**. Explicar las 6 fases en abstracto, sin
> datos sobre la mesa, no deja aprendizaje.

---

## Los seis campos de la fase 1 (los que hay que llenar antes de abrir Jupyter)

Esta es la operacionalización de la fase 1 que el curso usa. Ver detalle
en [`crisp-dm-fase1.md`](../s02-exploracion-polars/01-crisp-dm-fase1.md).

1. **Problema de negocio** (dos líneas, en lenguaje del dueño del proceso).
2. **Pregunta de negocio** (en voz del área que la hará).
3. **Objetivo de análisis** (lo que el equipo técnico va a construir).
4. **Criterio de éxito analítico** (Silhouette, AUC, F1, MAPE...).
5. **Criterio de éxito de negocio** (tasa de contacto, revenue, retención).
6. **Riesgos** (lo que puede impedir que llegue a producción).

Si una fila está en blanco, no se arranca. Esta es la regla que
distingue un proyecto bien formulado de uno que entrega un AUC 0,82 y
muere en el cajón.

---

## Para la clase

Esta página es lectura de **la primera semana**, antes de S02. No se
dicta "en clase": se asume que el estudiante llega habiéndola leído.
Durante S02 se cita como marco cuando aparece la fase 1.

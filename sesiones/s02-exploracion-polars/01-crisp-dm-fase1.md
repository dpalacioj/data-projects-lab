---
title: "01 · CRISP-DM Fase 1"
parent: "S02 · Exploración con Polars"
grand_parent: Sesiones
nav_order: 1
---
# CRISP-DM Fase 1 — Objetivo de análisis

> Spec: [`docs/specs/SPEC_segmentacion-clientes_polars.md`](../../docs/specs/SPEC_segmentacion-clientes_polars.md) · Cuaderno: [`00_objetivo_de_analisis.ipynb`](../../projects/00-segmentacion-clientes/notebooks/00_objetivo_de_analisis.ipynb).

## La pregunta

**¿Qué entregable concreto le vas a dejar al negocio al final del
módulo, y cómo mides que valió la pena construirlo?**

No es «hacer segmentación». Es «producir una ficha accionable por
segmento que marketing adopte».

## Punto de partida: el dolor

Pasa seguido: un equipo entrega un modelo con AUC 0,82 y seis meses
después nadie lo usa. La razón no suele ser técnica. Es que la ficha
analítica (métrica interna) nunca se tradujo a una ficha de negocio
(decisión por cliente, mes a mes).

CRISP-DM fase 1 (Chapman et al., 2000) existe para forzar esa traducción
**antes** de abrir Jupyter.

## Idea central

Rellenar seis campos, en papel, antes de tocar código:

1. **Problema de negocio** (dos líneas).
2. **Pregunta de negocio** (en voz del dueño del proceso).
3. **Objetivo de análisis** (lo que el equipo técnico va a construir).
4. **Criterio de éxito analítico** (Silhouette, AUC, F1, etc.).
5. **Criterio de éxito de negocio** (tasa de contacto, revenue, retención).
6. **Riesgos** (lo que puede impedir que llegue a producción).

Si una fila está en blanco, no se arranca.

## Teoría

CRISP-DM es cíclico: la fase 1 se **revisita** cuando la 2 (comprensión
de datos) o la 4 (modelado) descubren algo que invalida un supuesto.
Esto es sano, no un fracaso.

Referencias:
- **Chapman et al. (2000)**, _CRISP-DM 1.0 Step-by-step data mining
  guide_. SPSS / The CRISP-DM Consortium. (La fuente primaria; está en
  PDF y se cita desde ahí, no desde blogs.)
- **Shmueli et al.**, _Data Mining for Business Analytics_. Capítulos 1
  y 2 complementan con casos.

## Práctica

Caso Online Retail II:

| Campo | Contenido (cuaderno 00) |
|---|---|
| Problema de negocio | El retailer trata a todos los clientes por igual; las campañas de regalo rinden poco porque no hablan al cliente adecuado. |
| Pregunta de negocio | ¿Cómo agrupar a los clientes para que las acciones de marketing tengan rendimiento diferenciado? |
| Objetivo de análisis | Segmentos no supervisados sobre RFM extendido, estables entre K-Means y DBSCAN, con una ficha accionable por segmento. |
| Criterio analítico | Silhouette ≥ 0,35 y > 70 % de clientes concentrados en 3-5 segmentos interpretables. |
| Criterio de negocio | Marketing adopta las fichas (medible en U4 por tasa de contacto por segmento). |
| Riesgo principal | Que el pipeline no sea reproducible el día de grabación. **Mitigación**: `tests/test_cifras_referencia.py` y `scripts/figuras_u2.py` fallan si cambia alguna cifra. |

Tabla 2 completa en `notebooks/00_objetivo_de_analisis.ipynb`.

## Verificación

En clase, en voz alta: _«Si al final del módulo marketing nos pregunta
cómo probar que el modelo sirve, ¿con qué datos lo demostramos?»_

Si la respuesta no cita «tasa de contacto por segmento» o un número
derivable del artefacto `models/recompra_v1/metadatos.json`, la fila
«Criterio de negocio» no está bien escrita.

## Lo que **no** hagas

| Anti-patrón | Por qué | Qué hacer en su lugar |
|---|---|---|
| «Haremos segmentación» como objetivo | Es una técnica, no un objetivo | Describe la decisión que se tomará con la segmentación |
| Criterio analítico sin criterio de negocio | El modelo no se adopta | Siempre ambos; si uno falta, es una decisión diferida |
| Riesgos genéricos («calidad de datos») | No se puede mitigar lo abstracto | Riesgos atados a archivos del repo (parquet, metadatos.json) |

## Si falla

El estudiante no sabe cómo escribir el criterio de negocio:
→ [`07-troubleshooting.md`](07-troubleshooting.md) §«Mi criterio de negocio
no se puede medir».

## Siguiente

- Próximo tema: [`02-polars-vs-pandas.md`](02-polars-vs-pandas.md).
- Cuaderno asociado: [`00_objetivo_de_analisis.ipynb`](../../projects/00-segmentacion-clientes/notebooks/00_objetivo_de_analisis.ipynb).

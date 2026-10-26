---
title: "05 · Datos como código"
parent: Tutoriales
nav_order: 7
has_children: true
---
# Tutorial 02d — Datos como código: dataset card + contratos ejecutables

> Pre-requisito: tener el ambiente del curso montado ([tutorial 00](../00-ambientes-venv-conda-uv/)).
> Dependencia extra para este tutorial: `uv pip install pandera` (o `pip install pandera` si usas `.venv`).

Un modelo que entrena sin errores y sirve predicciones aparentemente razonables
**no es evidencia de que el dato sigue siendo el mismo**. Este tutorial muestra
cómo detectar silencios en los datos antes de que lleguen al modelo, con dos
disciplinas sencillas y durables:

1. **Dataset card** — documentar el dato con el mismo rigor con el que
   documentamos el modelo.
2. **Contratos de datos** ejecutables en tres niveles — reglas que el `pipeline`
   verifica antes de entrenar.

Este tutorial adapta la **sesión 2 de MLOps-Course** a un curso que todavía no
es de MLOps. Las herramientas pesadas (DVC, lakeFS, Delta/Iceberg) quedan
afuera; lo que se queda es la **disciplina**, aplicable desde el primer
proyecto.

---

## Mapa del tutorial

| Archivo | Qué hay dentro | Tiempo de lectura/ejecución |
|---|---|---|
| [`el-dolor-de-los-datos.ipynb`](el-dolor-de-los-datos.ipynb) | Notebook que entrena un modelo, cambia silenciosamente las unidades de una columna y muestra que ningún test estándar lo nota. Luego agrega un contrato de 3 niveles y lo captura. | 30 min |
| [`contratos-de-datos.md`](contratos-de-datos.md) | Los **tres niveles** de check (por fila, por distribución, entre columnas) con ejemplos en Pandera. Por qué el nivel 2 es el que casi nadie escribe. | 20 min |
| [`dataset-card.md`](dataset-card.md) | Ejemplo lleno de dataset card para **Online Retail II** (el caso guía del repositorio). Sirve de plantilla para la tuya. | 15 min |
| [`taller.md`](taller.md) | Aplicación individual: escribir la dataset card de **tu** dataset + 3 reglas de contrato (una por nivel). | 60 min |

---

## Orden sugerido

1. Abre el notebook [`el-dolor-de-los-datos.ipynb`](el-dolor-de-los-datos.ipynb) y
   recórrelo. Es el "problema primero". Si llegas al final y entiendes el
   mensaje, el resto del tutorial tiene propósito.
2. Lee [`contratos-de-datos.md`](contratos-de-datos.md). Es la teoría detrás del
   contrato que viste al final del notebook.
3. Lee [`dataset-card.md`](dataset-card.md) como ejemplo del formato esperado.
4. Haz el [`taller.md`](taller.md): tu propio dataset card + tus 3 reglas.

---

## Qué NO hace este tutorial

- **No** cubre versionado de datos (DVC, lakeFS, Delta). Esas son herramientas
  de MLOps y se ven en el curso siguiente.
- **No** cubre las tres formas de leakage temporal con profundidad. La [`sesión
  S03`](../../sesiones/s03-procesamiento-sklearn/) del módulo las aborda sobre
  el caso guía (`cohortes-y-fuga.md`).
- **No** reemplaza el EDA de la [`sesión S02`](../../sesiones/s02-exploracion-polars/).
  Es complementario: EDA te dice qué hay en el dato; el contrato te dice qué
  pasa si cambia.

---

## Qué sigue

Si terminaste el taller, puedes aplicar la disciplina a tu propio proyecto o
seguir con la [sesión S02](../../sesiones/s02-exploracion-polars/) del módulo,
donde la discusión de nulos estructurales vs. faltantes del cuaderno 01 encaja
directamente con lo que viste en los contratos de datos.

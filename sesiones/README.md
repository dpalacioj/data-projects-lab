---
title: Sesiones
nav_order: 2
has_children: true
---
# Sesiones del módulo Proyecto I

Módulo de la Especialización en Ciencia de Datos e IA (Universidad de
Medellín). Última revisión: 2026-10-19.

Cada sesión está organizada así: el `README.md` es la hoja de ruta
(encuadre, tiempos, bloques, cierre); los `.md` temáticos son el
material (teoría + práctica + verificación); `taller.md` es la
aplicación individual; `troubleshooting.md` responde a los errores
reales que ya vimos.

**Las sesiones no duplican código.** Enlazan a `projects/00-segmentacion-clientes/`
y a `tutorials/`. Cuando una cifra cambia en el proyecto, cambia en una
sola parte.

## Mapa del módulo

| Sesión | Unidad | Estado | Guión |
|---|---|---|---|
| **S01** · Encuadre del curso: CRISP-DM como columna vertebral | U1 | marco (no presencial) | [s01-objetivo-crisp-dm/](s01-objetivo-crisp-dm/) |
| **S02** · Exploración de datos con Polars | U2 | completa | [s02-exploracion-polars/](s02-exploracion-polars/) |
| **S03** · Procesamiento, pipelines y modelos | U3 | completa | [s03-procesamiento-sklearn/](s03-procesamiento-sklearn/) |
| **S04** · Aplicación: del modelo al entregable | U4 | marco (trabajo autónomo) | [s04-app-mlops/](s04-app-mlops/) |

**S01 y S04 son marco, no sesiones presenciales.** S01 la lees la primera
semana; mapea las 6 fases de CRISP-DM a dónde se practica cada una. S04
la trabajas al final del módulo sobre tu propia app, apoyado en el
proyecto [`projects/04-reconocimiento-imagenes/`](../projects/04-reconocimiento-imagenes/)
como ejemplo canónico.

**S02 y S03 son las sesiones presenciales con guión completo.**

### Cómo se mapean a las clases presenciales

El módulo son **8 clases presenciales de 4 horas** (dos por sesión
conceptual: U1 = clase 1 y 2, U2 = clase 3 y 4, U3 = clase 5 y 6,
U4 = clase 7 y 8). El calendario detallado con desglose de las 4h por
clase está en el [README raíz del repo](../README.md#calendario-detallado-8-clases--4-horas).

**Clase 8** es la presentación del entregable final; la especificación
de qué entregar está en [`trabajo-final/README.md`](../trabajo-final/README.md).

> Las sesiones completas (S02, S03) se congelaron contra el commit
> `main` de la fecha de última revisión. Si una cifra no coincide, el
> flujo es volver a correr los cuadernos del caso guía: el código
> reproduce; el repo no persigue cifras sueltas.

## Cómo leer una sesión

1. Abrir el `README.md` de la sesión. Es la hoja de ruta.
2. Mirar la tabla de bloques (duraciones y archivos asociados).
3. Para cada bloque, abrir el `.md` temático correspondiente y los
   cuadernos/scripts del proyecto que enlaza.
4. `taller.md` → aplicación individual; `troubleshooting.md` →
   consulta cuando algo falla.

# Sesiones del módulo Proyecto I

> Revisión: **2026-10-19** · Versión del repo: ver `git log main..HEAD` · Módulo: Especialización en Ciencia de Datos e IA (Universidad de Medellín).

Cada sesión es una **guía para el instructor**: el `README.md` de la
sesión es el guión (encuadre, tiempos, bloques, cierre); los `.md`
temáticos son el material (teoría + práctica + verificación); `taller.md`
es la aplicación independiente; `troubleshooting.md` responde a los
errores reales que ya vimos; `checklist-dia-de-clase.md` es lo que se
revisa 15 minutos antes de entrar al aula.

**Las sesiones no duplican código.** Enlazan a `projects/00-segmentacion-clientes/`,
a `tutorials/` y a `docs/specs/`. Cuando una cifra cambia en el proyecto,
cambia en una sola parte.

## Mapa del módulo

| Sesión | Unidad | RAE | Estado | Guión |
|---|---|---|---|---|
| **S01** · Encuadre del curso: CRISP-DM como columna vertebral | U1 | 1.1 | 📄 marco (no presencial) | [s01-objetivo-crisp-dm/](s01-objetivo-crisp-dm/) |
| **S02** · Exploración de datos con Polars | U2 | 1.1 | 🟢 completa | [s02-exploracion-polars/](s02-exploracion-polars/) |
| **S03** · Procesamiento, pipelines, modelos y LLM | U3 | 2.1 | 🟢 completa | [s03-procesamiento-sklearn/](s03-procesamiento-sklearn/) |
| **S04** · Aplicación: del modelo al entregable | U4 | 3.1/4.1 | 📄 marco (trabajo autónomo) | [s04-app-mlops/](s04-app-mlops/) |

> **S01 y S04 son marco, no sesiones presenciales.** S01 la lees la
> primera semana; mapea las 6 fases de CRISP-DM a dónde se practica cada
> una. S04 la trabajas al final del módulo sobre tu propia app, apoyado
> en el proyecto [`projects/04-reconocimiento-imagenes/`](../projects/04-reconocimiento-imagenes/)
> como ejemplo canónico.
>
> **S02 y S03 son las sesiones presenciales con guión completo.**

> Las sesiones completas (S02, S03) se grabaron contra el commit `main`
> del día de la grabación. Si una cifra no coincide, se re-captura la
> **spec** correspondiente en `docs/specs/`, no la prueba del repo (ver
> `_templates/CONVENCIONES.md`).

## Cómo leer una sesión

1. Abrir el `README.md` de la sesión. Es el guión.
2. Mirar la tabla de bloques (duraciones y archivos asociados).
3. Para cada bloque, abrir el `.md` temático correspondiente y los
   cuadernos/scripts del proyecto que enlaza.
4. `taller.md` → aplicación individual; `troubleshooting.md` →
   consulta cuando algo falla.

## Para crear una sesión nueva

1. Copiar `_templates/GUION_SESION.md` a `sXX-nombre/README.md`.
2. Para cada tema, copiar `_templates/TEMA.md` a `sXX-nombre/<tema>.md`.
3. Llenar `taller.md`, `troubleshooting.md`, `checklist-dia-de-clase.md`.
4. Actualizar la tabla de arriba.

Las convenciones de estilo, naming y commits están en
[`_templates/CONVENCIONES.md`](_templates/CONVENCIONES.md).

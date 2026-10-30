# Sesión SXX · «Nombre de la sesión»

Última revisión: YYYY-MM-DD. Versiones verificadas: Python 3.11,
_(añadir las que apliquen)_. La sesión implementa
[`docs/specs/<SPEC_...>.md`](../../docs/specs/) y se apoya en
[`projects/<proyecto>/`](../../projects/).

La pregunta que organiza la sesión es:

> _(Formula aquí la pregunta fundacional en una línea. No describas la
> herramienta; describe lo que el alumno debe PODER responder al terminar.)_

---

## 0. Para antes de clase

- [ ] Verificar la [`checklist-dia-de-clase.md`](checklist-dia-de-clase.md).
- [ ] Correr `uv sync --extra <extra>` y los smoke tests indicados.
- [ ] Abrir los cuadernos del proyecto y asegurarse de que ejecutan.

## 1. Objetivos de aprendizaje

Al terminar la sesión el estudiante puede:

1. _(Verbos medibles: construir, diferenciar, decidir, justificar.)_
2. ...
3. ...

**Autovalidación** (sección 6): si no respondes a las 5 preguntas finales,
vuelve al bloque correspondiente.

## 2. Organización temporal

| Bloque | Duración | Contenido | Documento |
|---|---|---|---|
| **El problema** | ~10 min | _(experiencia concreta del dolor, antes de herramientas)_ | `<notebook o enlace>` |
| **A — <Tema 1>** | X–Y min | _(resumen del bloque)_ | [`<tema1>.md`](<tema1>.md) |
| **B — <Tema 2>** | X–Y min | _(resumen del bloque)_ | [`<tema2>.md`](<tema2>.md) |
| **Taller** | ~Z min | Aplicación independiente | [`taller.md`](taller.md) |

## 3. El problema (antes de las herramientas)

_(En voz alta, antes de abrir cualquier IDE. Tres preguntas que incomodan
al estudiante y motivan los bloques siguientes. Usa un ejemplo CONCRETO
del proyecto; no abstracto.)_

## 4. Materiales y artefactos

- [`<tema1>.md`](<tema1>.md) — _(propósito del documento en una línea)_
- [`<tema2>.md`](<tema2>.md) — ...
- `taller.md` — aplicación.
- `troubleshooting.md` — errores reales ya documentados.
- Cuadernos del proyecto: `projects/<proyecto>/notebooks/<XX_...>.ipynb`.
- Módulos del paquete: `projects/<proyecto>/src/<paquete>/<...>.py`.

## 5. Quick start (macOS/Linux)

```bash
# desde la raíz del repositorio
uv sync --extra <extra>
cd projects/<proyecto>
uv run jupyter lab
```

> Windows: ver [`troubleshooting.md`](troubleshooting.md) sección
> «ExecutionPolicy antes del primer script».

## 6. Autoverificación

Cinco preguntas sin spoiler. Dónde buscar la respuesta entre paréntesis.

1. _(pregunta 1)_ (ver `<tema1>.md` §X).
2. _(pregunta 2)_ (ver `<tema2>.md` §Y).
3. ...

## 7. Alternativas con criterio

Lista las opciones que considerábamos y por qué elegimos la que elegimos.
Si cambia la fecha de referencia, revisar.

| Alternativa | Cuándo elegirla | Qué dejas en la mesa |
|---|---|---|
| ... | ... | ... |

## 8. Anti-patrones

Lo que **no** hagas y por qué. Idealmente, errores que ya vimos.

| No hagas | Por qué | Qué hacer en su lugar |
|---|---|---|
| ... | ... | ... |

## 9. Cierre y puente

- **Tarea:** _(un solo entregable claro, con fecha de corte si aplica)_.
- **Puente a S0X+1:** _(una línea que anticipa el próximo problema, no
  la próxima herramienta)_.

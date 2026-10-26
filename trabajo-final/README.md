---
title: Trabajo final
nav_order: 5
---
# Trabajo final del módulo Proyecto I

Este documento describe **qué se espera que entregues** al final del
módulo. No es la rúbrica detallada que verá el equipo docente; es lo
que tú, como estudiante, debes construir y por qué.

Última revisión: **2026-10-25**

---

## En una frase

Entregar un **proyecto de ciencia de datos funcional, modularizado y
reproducible**, aplicado a un problema real que te interese, con el
mismo esqueleto que [`projects/05-ejemplo-completo/`](../projects/05-ejemplo-completo/).

Si te preguntan en una entrevista *"muéstrame un proyecto tuyo"*, este
es el que mandas.

---

## Qué entregar

Un repositorio en GitHub (público o privado, con los profesores
invitados) que contenga:

### 1. Un README que puedas defender

Alguien que no te conoce lo lee y debe poder responder:

- ¿Qué problema resuelve este proyecto?
- ¿De dónde salieron los datos?
- ¿Cómo lo corro?
- ¿Qué resultado puedo esperar?

Si tu README no responde esas 4 preguntas, no te lo acepto.

### 2. Datos documentados

- Fuente real (una URL, un dataset público, una muestra de tu trabajo).
- Un `dataset-card.md` corto (ver [`tutorials/05-datos-como-codigo/`](../tutorials/05-datos-como-codigo/)) que diga: qué contiene cada columna, de dónde viene, con qué licencia, cuántas filas, qué filas son filtradas y por qué.
- Los datos crudos **no se versionan** en git. Un script los descarga o los genera.

### 3. Estructura de código modular

Tu repositorio debe seguir este patrón mínimo (misma plantilla que el
proyecto 05):

```
tu-proyecto/
├── README.md
├── Makefile                     ← comandos del proyecto
├── pyproject.toml               ← dependencias con uv
├── .github/workflows/ci.yml     ← integración continua
│
├── notebooks/                   ← experimentación (no producción)
├── src/<tu_paquete>/            ← código modular
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── app/                         ← Streamlit o FastAPI (al menos uno)
├── tests/                       ← pruebas automatizadas
└── models/                      ← modelos entrenados (gitignored si son grandes)
```

**No puede faltar** `src/`, `tests/`, `Makefile`, y `.github/workflows/ci.yml`.

### 4. Al menos un entregable servido

Elige uno (o ambos):

- **Streamlit**: formulario que recibe inputs del usuario y muestra la predicción.
- **FastAPI**: endpoint `POST /predict` que recibe JSON y devuelve JSON.

Si eliges FastAPI, documenta la API (FastAPI te la genera gratis en `/docs`).

### 5. Tests que pasen en CI

- Mínimo 3 pruebas automatizadas:
  - Una verifica la **carga del dataset**.
  - Una verifica el **pipeline de features**.
  - Una verifica la **inferencia end-to-end** (entrena un modelo chico, predice, verifica que no explota).
- Un `.github/workflows/ci.yml` que las corre en cada push y PR.
- **El último commit de tu rama debe tener check verde ✅.** Si está en rojo, no se evalúa.

### 6. Commits coherentes

- Mensajes en formato **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Al menos 10 commits en el historial. No se acepta un repositorio con
  un solo commit "subida final".
- Al menos un Pull Request mergeado a `main` (puede ser contra ti mismo).

---

## Cómo arrancar

1. **Elige un problema.** Que te interese genuinamente. Si es aburrido,
   el proyecto será aburrido. Clasificación o regresión. Datos
   tabulares, texto o imágenes.
2. **Encuentra los datos.** Kaggle, UCI, portales de gobierno, APIs
   públicas. Guarda el link en tu `dataset-card.md`.
3. **Haz CRISP-DM fase 1 primero.** Antes de abrir Jupyter: los 6 campos
   que están en [`sesiones/s01-objetivo-crisp-dm/`](../sesiones/s01-objetivo-crisp-dm/).
4. **Clona la plantilla.** Copia la estructura de
   [`projects/05-ejemplo-completo/`](../projects/05-ejemplo-completo/)
   a un repositorio nuevo. Renombra el paquete, cambia la fuente de
   datos, deja todo lo demás.
5. **Trabaja en ramas.** `feat/exploracion`, `feat/pipeline`, `feat/app`,
   etc. Un PR por cada hito.
6. **Pide revisión temprana.** No esperes a la última semana. Abre un PR
   draft y pide feedback.

---

## Criterios de evaluación

Se evalúa en cuatro dimensiones. Cada una vale ~25%:

### Reproducibilidad (25%)

- El proyecto corre con `make setup && make train && make test` sin pedir ayuda.
- `.gitignore` tiene lo que no debe commitearse (datos crudos, modelos grandes, `.env`).
- Un revisor en una máquina limpia puede clonar y correr en < 10 minutos.

### Modularidad (25%)

- `src/` tiene **un archivo por responsabilidad**. Nada de un `main.py` gigante.
- Las apps (`app/`) son delgadas: solo importan y llaman a `src/`.
- Los notebooks son **experimentación**, no contienen lógica que la app ejecute.

### Honestidad técnica (25%)

- Las métricas se miden **sin fuga de datos** (split antes de preprocesar).
- Si tus datos tienen componente temporal, usas split temporal (ver [`projects/d2-temporal-split/`](../projects/d2-temporal-split/)).
- El README cuenta las **limitaciones del modelo**, no solo lo bueno.

### Entregable (25%)

- La app o la API funcionan. Un revisor abre la UI o llama al endpoint y recibe una predicción coherente.
- Hay un README que un extraño entiende.
- CI está verde en el commit evaluado.

---

## Lo que NO se evalúa

- La métrica absoluta del modelo. Un AUC 0.72 bien justificado vale más
  que un AUC 0.95 con fuga.
- La complejidad del modelo. Un `LogisticRegression` bien entrenado vale
  igual que una red neuronal, si el problema es simple.
- La elegancia del diseño. Simple y claro vale más que elegante y oscuro.

---

## Recursos del repo que vas a usar

| Recurso | Para qué |
|---|---|
| [`projects/05-ejemplo-completo/`](../projects/05-ejemplo-completo/) | La plantilla. Cópiala y adapta |
| [`tutorials/01-git-flujo-de-trabajo.md`](../tutorials/01-git-flujo-de-trabajo.md) | Si el flujo de ramas y PR no te es claro |
| [`tutorials/01b-tu-primer-ci.md`](../tutorials/01b-tu-primer-ci.md) | Para armar tu propio CI desde cero |
| [`tutorials/02-configuracion-proyecto-uv.md`](../tutorials/02-configuracion-proyecto-uv.md) | Estructura con uv |
| [`tutorials/03-el-makefile-del-curso.md`](../tutorials/03-el-makefile-del-curso.md) | Qué poner en el Makefile |
| [`tutorials/06-del-notebook-al-paquete/`](../tutorials/06-del-notebook-al-paquete/) | La migración notebook → módulos, paso a paso |
| [`tutorials/07-tecnicas-preprocesamiento.md`](../tutorials/07-tecnicas-preprocesamiento.md) | Pipeline sin fuga, imputación, encoding |
| [`tutorials/09-metricas-ml-regresion-vs-clasificacion.md`](../tutorials/09-metricas-ml-regresion-vs-clasificacion.md) | Qué métrica usar para tu problema |
| [`tutorials/12-mlflow-y-fastapi-para-principiantes.md`](../tutorials/12-mlflow-y-fastapi-para-principiantes.md) | Si quieres añadir tracking de experimentos |

---

## Preguntas frecuentes

**¿Puedo usar un dataset que ya haya trabajado en otra clase?**
Sí, siempre que el enfoque sea distinto y el proyecto sea tuyo de principio a fin.

**¿Puedo trabajar en grupo?**
Sí, hasta 3 personas. Cada persona debe tener al menos 3 PRs mergeados a mano con commits propios (no co-authored dummy).

**¿Puedo usar deep learning?**
Sí, si el problema lo justifica. Mira [`projects/04-reconocimiento-imagenes/`](../projects/04-reconocimiento-imagenes/) como referencia para el flujo con PyTorch.

**¿Qué pasa si mi modelo tiene accuracy baja?**
No se penaliza. Lo que se penaliza es que no sepas por qué.

**¿Dónde entrego?**
Pull Request abierto en tu repositorio contra `main` con el link al PR enviado al profesor. El commit a evaluar es el que esté verde en CI al momento del cierre.

---

## Fecha de entrega

Clase 8 (última sesión presencial). Presentación de 15 minutos por
estudiante o grupo.

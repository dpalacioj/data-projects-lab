# Convenciones del repositorio

Última revisión: 2026-10-19. Si algo aquí contradice una spec de
`docs/specs/`, **manda la spec**.

## Idioma

- **Español** en: código (nombres de funciones y variables de dominio),
  docstrings, cuadernos, mensajes de commit, documentación narrativa.
- **Inglés** en: identificadores que replican una API externa (p.ej.
  `random_state`, `n_jobs`), palabras clave reservadas, términos
  técnicos sin traducción consensuada (ROC, AUC, CV).

## Fechas

- Toda guía (`.md`), ficha de modelo (`metadatos.json`), spec (`docs/specs/`)
  y resumen de resultados declara **fecha** y **versiones verificadas**
  (Python, Polars, scikit-learn, Pydantic).
- Cuando un número se vuelve «cita» (p. ej. `AUC = 0,785`), se registra
  en una spec o en un `HALLAZGOS.md`. **El código no inventa número; el
  código reproduce.**

## Fuente de verdad

- **Una** por tipo de artefacto. Si existen un script y un cuaderno que
  calculan lo mismo, uno importa del otro; no se repite el cálculo.
- Las cifras de referencia viven en la spec (`docs/specs/`) y las
  verifica una prueba `@pytest.mark.slow` del repo.

## Reproducibilidad

- **Semilla única por proyecto**: `SEMILLA = 42` en `config.py`. Alias
  `RANDOM_STATE` solo donde una API externa lo exija.
- **Orden fijo de filas**: tras cualquier `polars.unique()`, un
  `.sort(<clave>)` explícito. El bosque entrenado dos veces debe producir
  predicciones idénticas (ver `tests/test_reproducibilidad.py`).
- **Fit solo con el entrenamiento**: todo `fit` ocurre dentro del
  `Pipeline` y después de la partición. Verificado en
  `tests/pipelines/test_sin_fuga.py`.

## Fugas de datos: la lista

1. Variables calculadas con información posterior al corte.
2. Preprocesadores ajustados sobre A ∪ B.
3. Métricas sobre el conjunto que usamos para elegir hiperparámetros.
4. Imputación por media global antes de partir.
5. `OneHotEncoder` sin `handle_unknown="ignore"` que falla en inferencia.

## Dependencias

- `uv` como gestor. `uv.lock` versionado.
- `scikit-learn>=1.9`, `polars>=2.0`, `pydantic>=2.0` para
  `segmentacion-clientes`. Si un extra existente usa versiones distintas
  (p. ej. `meli` fija pydantic v1), **no se reutiliza**; se declara el
  nuevo pin.

## Commits

- Convencional (`feat`, `fix`, `docs`, `test`, `chore`, `refactor`,
  `tutorial`, `data`, `perf`) con alcance opcional
  (`feat(features): ...`).
- Mensaje en español. Primera línea ≤ 72 caracteres. Cuerpo con el
  **por qué** y el **cómo se verificó**.
- Una idea por commit; `uv run pytest` verde al final del commit.

## Cuadernos

- Prohibido `!pip` y `!wget` en celdas. Instalación por `uv sync`;
  descarga por `scripts/descargar_datos.py` con hash.
- Salidas no se commitean (hook `nbstripout`).
- Imports a `segmentacion.*` (nunca `src.*`).
- Toda celda que llama al disco debe tener un `assert` que falle con
  mensaje claro si el archivo no existe.

## Modelos

- Artefacto en `models/<nombre>_v<N>/`: `pipeline.joblib` (gitignored) +
  `metadatos.json` (versionado) con los 10 campos de la Tabla 7 de la spec
  de la Unidad 3. `cargar()` falla con `VersionIncompatible` si cambia la versión de
  scikit-learn.

## LLMs

- Cliente por defecto: `FakeLLM`. Tests y cuadernos corren sin red ni
  claves. Caché versionada en `data/interim/`.
- Un único adaptador real: `cliente_openai_compatible` por variables de
  entorno (ver `.env.example`). **Ningún archivo nombra un proveedor
  fuera del adaptador y del README.**

## Ruff y pre-commit

- Ruff format cubre `projects/00-segmentacion-clientes/**` (no formatea el
  resto para no entrar en lint pre-existente). Mismo criterio para
  `nbstripout`: solo cuadernos del proyecto U2+U3.
- Pre-commit obligatorio: `uv run pre-commit install` tras clonar.

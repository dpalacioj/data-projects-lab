---
title: "06 · Taller"
parent: "S03 · Procesamiento, pipelines, modelos"
grand_parent: Sesiones
nav_order: 6
---
# Taller S03 — Aplicación independiente

> Entregable: ver §4. Tiempo estimado: 8-12 horas de trabajo autónomo.

## Reglas del taller

1. Dataset propio con eje temporal (debe ser **razonable** partir por
   fecha). Si en S02 usaste un dataset temporal, puedes continuar con él.
2. Problema **supervisado binario** (puede ser clasificación o
   predicción binaria de un evento futuro).
3. Mínimo dos cohortes temporales (A y B) y una regla por recencia (u
   otra variable observable) como línea base.
4. Pipeline sin fuga obligatorio (verificable con un test).
5. Modelo serializado con `metadatos.json` siguiendo los 10 campos
   obligatorios de la ficha del caso guía.

## Entrega

Pull request contra tu rama o repositorio propio con:

- `notebooks/06_particion_y_fuga.ipynb` — cohortes A y B; mínimo una
  demostración de fuga.
- `notebooks/07_ingenieria_caracteristicas.ipynb` — `ColumnTransformer`
  con bloques explicables (al menos `NUM_LOG`, `NUM_STD`, `CAT`).
- `notebooks/09_modelos_y_pipelines.ipynb` — tres modelos + línea base
  + tabla de umbrales + importancia de variables.
- `notebooks/10_segmentacion_no_supervisada.ipynb` — opcional, si tu
  dominio lo permite.
- `src/<paquete>/features/{devoluciones?, etiquetas, cohortes}.py` — al
  menos los módulos de etiqueta y cohorte.
- `src/<paquete>/pipelines/preprocesamiento.py` — con tus constantes
  `NUM_LOG`, `NUM_STD`, `CAT`.
- `src/<paquete>/models/{entrenar, registro}.py` — tu `MODELOS` dict y
  `guardar()`/`cargar()`.
- `src/<paquete>/evaluation/{metricas, umbral}.py`.
- `tests/features/`, `tests/pipelines/`, `tests/models/`,
  `tests/evaluation/` — una prueba por módulo con ≤ 20 filas.
- `tests/test_cifras.py` marcado `@pytest.mark.slow` — reproduce las
  cifras clave de tu caso.
- `tests/test_reproducibilidad.py` — tu modelo entrenado dos veces da
  predicciones idénticas (`n_jobs=1`).
- `tests/pipelines/test_sin_fuga.py` — scaler aprende las medias de A,
  no de A ∪ B.
- `models/<nombre>_v1/metadatos.json` versionado.
- `scripts/caso_<tu_dominio>.py` — envoltorio reproducible.

## Criterios de aceptación

- [ ] `uv run pytest` verde en < 15 s (sin el `-m slow`).
- [ ] `uv run pytest -m slow` verde (puede durar más).
- [ ] `tests/pipelines/test_sin_fuga.py` existe y pasa.
- [ ] `tests/test_reproducibilidad.py` existe y pasa (igualdad exacta
      con `n_jobs=1`).
- [ ] `models/<nombre>_v1/metadatos.json` tiene los 10 campos
      obligatorios (fecha_corte, horizonte, variables_en_orden,
      semilla, versiones, datos, auc_validacion, umbral_acordado,
      fecha, autor).
- [ ] `cargar()` falla con mensaje claro si cambia la versión de
      scikit-learn.
- [ ] Ningún `!pip` ni `!wget` en celdas.
- [ ] Línea base reportada; el modelo adoptado supera a la línea base
      en AUC y, idealmente, en F1.

## Rúbrica

| Dimensión | Peso | Qué miramos |
|---|---|---|
| Cohortes y fuga | 20 % | Dos cohortes temporales, al menos una demostración de fuga |
| Pipeline sin fuga | 20 % | `test_sin_fuga` pasa; medias del scaler = medias de A |
| Línea base | 10 % | Regla simple reportada; comparación honesta con el modelo |
| Modelos + umbrales | 20 % | Tres modelos; tabla de umbrales; umbral acordado |
| Artefacto + registro | 15 % | `metadatos.json` completo; `cargar()` falla con versión distinta |
| Reproducibilidad | 10 % | `test_reproducibilidad` pasa |
| Narrativa | 5 % | README, HALLAZGOS actualizado, decisiones documentadas |

## Bloque LLM (opcional)

Si tu dominio incluye texto (nombres de producto, descripciones,
comentarios), aplica el patrón del cuaderno 08:

- 12 categorías cerradas de tu dominio.
- Pydantic como esquema.
- `FakeLLM` por defecto para los tests.
- Un gold set de 100-200 filas etiquetadas a mano.
- Regla por palabras clave como línea base.
- Comparación con y sin las columnas del LLM en el modelo final.

Bonus en la rúbrica (+10 %) si el bloque LLM está completo con
pruebas sin red.

## Entregables opcionales

- Modelo adicional (XGBoost, LightGBM) comparado contra los tres del
  curso.
- Análisis de drift: AUC esperado vs. AUC observado en un conjunto
  posterior a B.
- Figuras análogas a `g_fuga.png`, `g_sobreajuste.png`, `g_roc_pr.png`,
  `g_importancia.png` generadas desde tu caso.

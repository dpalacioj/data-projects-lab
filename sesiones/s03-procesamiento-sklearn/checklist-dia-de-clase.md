# Checklist S03 — 15 minutos antes de entrar al aula

## Entorno

- [ ] `uv --version` ≥ 0.9.0.
- [ ] `uv sync --extra segmentacion-clientes --extra dev` sin errores.
- [ ] `uv run python -c "import polars, sklearn, pydantic; print(sklearn.__version__)"`
      imprime `1.9.1` (o la versión acordada).

## Datos

- [ ] `projects/segmentacion-clientes/data/raw/online_retail_II.parquet`
      existe (requisito de S02; si no, correr `scripts/descargar_datos.py`).
- [ ] `projects/segmentacion-clientes/data/processed/ventas_validas.parquet`
      y `.../cancelaciones.parquet` existen (los deja el cuaderno 02 de S02).

## Pruebas rápidas (sin disco)

- [ ] `uv run pytest projects/segmentacion-clientes/tests/features/
      projects/segmentacion-clientes/tests/pipelines/
      projects/segmentacion-clientes/tests/models/
      projects/segmentacion-clientes/tests/evaluation/` verde en < 10 s.

## Pruebas slow (reproducen §9)

- [ ] `uv run pytest -m slow
      projects/segmentacion-clientes/tests/test_cifras_recompra.py` verde
      (20 asserts; tolera 5e-3 en 4 cifras del bosque, ver docstring).

## Scripts

- [ ] `uv run python projects/segmentacion-clientes/scripts/caso_recompra.py`
      imprime las cohortes A/B y las métricas de los 3 modelos.
- [ ] `uv run python projects/segmentacion-clientes/scripts/figuras_u3.py`
      deja 4 PNG en `reports/figures/g_*.png`.

## Cuadernos

- [ ] `uv run jupyter nbconvert --to notebook --execute
      projects/segmentacion-clientes/notebooks/06_particion_y_fuga.ipynb
      --output /tmp/nb06.ipynb` pasa.
- [ ] Igual para 07, 09, 10.

## Artefacto

- [ ] `projects/segmentacion-clientes/models/recompra_v1/metadatos.json`
      existe y los 10 campos están llenos.
- [ ] `uv run python -c "
      import sys; sys.path.insert(0, 'projects/segmentacion-clientes/src')
      from segmentacion.models.registro import cargar
      from pathlib import Path
      pipe, meta = cargar(Path('projects/segmentacion-clientes/models/recompra_v1'))
      print('ok', meta['umbral_acordado'])"` imprime `ok 0.3`.

## Si vas a enseñar el bloque LLM

- [ ] `uv run pytest
      projects/segmentacion-clientes/tests/features/test_texto_llm.py
      projects/segmentacion-clientes/tests/features/test_llm_clientes.py`
      verde (10 asserts).
- [ ] `uv run jupyter nbconvert --execute
      projects/segmentacion-clientes/notebooks/08_llm_como_transformador.ipynb
      --output /tmp/nb08.ipynb` con `FakeLLM`.
- [ ] Decisión: ¿clase con `FakeLLM` o con cliente real?
    - Con `FakeLLM`: no requiere `.env`; demostración didáctica.
    - Con cliente real: requiere `.env` con `LLM_BASE_URL`, `LLM_API_KEY`,
      `LLM_MODELO`. Verifica consumo esperado.

## Narrativa

- [ ] [`HALLAZGOS.md`](../../projects/segmentacion-clientes/HALLAZGOS.md)
      abierto (se referencia como contrato con U2).
- [ ] `models/recompra_v1/metadatos.json` abierto (se referencia en el
      bloque D).
- [ ] Guión de la sesión abierto (segunda pestaña).

## Fallback sin internet

Si todo lo anterior pasó:

- Los cuadernos ya ejecutan desde el Parquet local.
- El pipeline del bosque está entrenado y serializado.
- La única caída posible es el cliente LLM real; para la clase usar
  `FakeLLM` (sin red).

## Después de clase

- [ ] Si editaste `HALLAZGOS.md`, commitea.
- [ ] Si re-entrenaste el modelo, confirma que `metadatos.json` tiene
      la nueva fecha + autor.
- [ ] Si corriste el cliente real, confirma que la caché actualizada
      (`data/interim/categorias_llm.parquet`) se commitea.

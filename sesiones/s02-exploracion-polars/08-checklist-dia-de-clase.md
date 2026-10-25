# Checklist S02 — 15 minutos antes de entrar al aula

> El objetivo es que, si la red se cae, puedas seguir con lo que tienes en disco.

## Entorno

- [ ] `uv --version` ≥ 0.9.0.
- [ ] `uv sync --extra segmentacion-clientes --extra dev` sin errores
      (toma ~30 s si ya hay caché).
- [ ] `uv run python -c "import polars, pydantic, data_profiling, skimpy; print('OK')"`.

## Datos

- [ ] `projects/00-segmentacion-clientes/data/raw/online_retail_II.parquet`
      existe (6,5 MB). Si no:
      ```bash
      uv run python projects/00-segmentacion-clientes/scripts/descargar_datos.py
      ```
- [ ] Hash del zip coincide con `SHA256_ZIP` en `descargar_datos.py`
      (el script lo verifica al correr; si falla, re-captura).

## Pruebas

- [ ] `uv run pytest projects/00-segmentacion-clientes/tests/test_limpieza.py
      projects/00-segmentacion-clientes/tests/test_rfm.py` verde en < 5 s.
- [ ] `uv run pytest -m slow projects/00-segmentacion-clientes/tests/test_cifras_referencia.py`
      verde en < 10 s (32 asserts).

## Cuadernos

- [ ] `uv run jupyter nbconvert --to notebook --execute
      projects/00-segmentacion-clientes/notebooks/01_comprension_datos.ipynb
      --output /tmp/nb01.ipynb` pasa.
- [ ] Igual para `02_limpieza`, `04_eda`, `03_variables_cliente`,
      `05_eda_automatizado`.

## Figuras y perfil

- [ ] `uv run python projects/00-segmentacion-clientes/scripts/figuras_u2.py
      projects/00-segmentacion-clientes/data/raw/online_retail_II.parquet
      projects/00-segmentacion-clientes/reports/figures` deja 4 PNG.
- [ ] `uv run python projects/00-segmentacion-clientes/scripts/perfil.py`
      deja `reports/perfil_muestra.html` (2,2 MB).

## Narrativa

- [ ] `HALLAZGOS.md` abierto en el editor (lo editamos en vivo si surge
      un hallazgo).
- [ ] Guión de la sesión (`README.md`) abierto en segunda pestaña.

## Fallback sin internet

Si todo lo anterior pasó, puedes dictar la sesión sin red:
- Los cuadernos ya están ejecutados en memoria.
- `uv run jupyter lab` funciona sin internet.
- La única caída posible sería si el estudiante pide re-descargar el
  dataset; en ese caso, úsalo desde el Parquet local (ya está).

## Después de clase

- [ ] Guardar los `HALLAZGOS.md` nuevos (si editaste en vivo).
- [ ] Si la sesión era grabada, apagar el `jupyter lab` **antes** de
      cerrar la grabación (para evitar que el navegador reconecte).

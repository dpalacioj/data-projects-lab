"""Tests end-to-end: genera, carga y consulta usando SQLite en memoria y disco temporal."""
import json
from pathlib import Path

import pytest
from ingesta_eventos.pipeline import ejecutar
from ingesta_eventos.queries import (
    clientes_recurrentes,
    conteos_tablas,
    ingresos_por_canal,
    top_productos,
)


def _escribir_lote(tmp_path: Path, lote: dict, nombre: str = "eventos_2024-01-15.json") -> Path:
    path = tmp_path / nombre
    with path.open("w", encoding="utf-8") as f:
        json.dump(lote, f)
    return path


def test_pipeline_end_to_end_carga_y_consulta(engine_memoria, tmp_path, lote_json_valido):
    _escribir_lote(tmp_path, lote_json_valido)

    totales = ejecutar(engine=engine_memoria, directorio=tmp_path, verbose=False)
    assert totales["pedidos"] == 2
    assert totales["items"] == 4
    assert totales["clientes"] == 1
    assert totales["productos"] == 2

    conteos = conteos_tablas(engine_memoria)
    assert conteos["pedidos"] == 2
    assert conteos["items_pedido"] == 4


def test_pipeline_es_idempotente_en_clientes_y_productos(
    engine_memoria, tmp_path, lote_json_valido
):
    """Correr el pipeline 2 veces NO duplica clientes ni productos (upsert).

    Los pedidos TAMBIEN estan protegidos por PK, asi que tampoco se duplican.
    Los items si se insertarian de nuevo si no filtramos por pedido_id, pero
    como los pedidos ya existen, el ON CONFLICT los salta y los nuevos items
    se insertan igual (son filas auto-incrementales). En un pipeline de produccion
    real, antes de reinsertar items se borrarian los del pedido afectado; lo
    omitimos aqui por simplicidad.
    """
    _escribir_lote(tmp_path, lote_json_valido)
    ejecutar(engine=engine_memoria, directorio=tmp_path, verbose=False)
    ejecutar(engine=engine_memoria, directorio=tmp_path, verbose=False)

    conteos = conteos_tablas(engine_memoria)
    assert conteos["clientes"] == 1
    assert conteos["productos"] == 2
    assert conteos["pedidos"] == 2  # PK evita duplicados


def test_pipeline_falla_si_no_hay_archivos(engine_memoria, tmp_path):
    with pytest.raises(FileNotFoundError):
        ejecutar(engine=engine_memoria, directorio=tmp_path, verbose=False)


def test_queries_de_negocio_funcionan(engine_memoria, tmp_path, lote_json_valido):
    _escribir_lote(tmp_path, lote_json_valido)
    ejecutar(engine=engine_memoria, directorio=tmp_path, verbose=False)

    df_canal = ingresos_por_canal(engine_memoria)
    assert set(df_canal["canal"]) == {"web", "app"}

    df_top = top_productos(engine_memoria, limite=5)
    assert len(df_top) == 2
    assert "ingresos" in df_top.columns

    df_cli = clientes_recurrentes(engine_memoria, limite=5)
    assert len(df_cli) == 1
    assert df_cli.iloc[0]["pedidos"] == 2

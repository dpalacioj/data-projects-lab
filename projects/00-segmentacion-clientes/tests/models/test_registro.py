"""Tests de `segmentacion.models.registro`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from segmentacion.models.registro import (
    MetadatosIncompletos,
    VersionIncompatible,
    cargar,
    guardar,
)
from segmentacion.pipelines.preprocesamiento import construir_preprocesador
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _pipe_entrenado(seed: int = 42) -> Pipeline:
    rng = np.random.default_rng(seed)
    n = 40
    X = pd.DataFrame(
        {
            "monetario": rng.uniform(10, 1000, n),
            "frecuencia": rng.integers(1, 10, n),
            "productos_distintos": rng.integers(1, 50, n),
            "recencia": rng.uniform(0, 500, n),
            "antiguedad": rng.uniform(0, 800, n),
            "tasa_devolucion": rng.uniform(0, 0.3, n),
            "pais": rng.choice(["UK", "otro"], n),
        }
    )
    y = (X["recencia"] < X["recencia"].median()).astype(int)
    pipe = Pipeline(
        [
            ("pre", construir_preprocesador()),
            ("mod", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _metadatos_validos() -> dict:
    return {
        "fecha_corte": "2011-06-01",
        "horizonte": 90,
        "variables_en_orden": ["monetario", "recencia", "pais"],
        "semilla": 42,
        "versiones": {"python": "3.11", "polars": "2.0"},  # sklearn se autocompleta
        "datos": {"ruta": "data/raw/online_retail_II.parquet", "hash": "abc123"},
        "auc_validacion": 0.817,
        "umbral_acordado": 0.3,
        "fecha": datetime(2026, 10, 16).isoformat(),
        "autor": "David",
    }


def test_guardar_y_cargar_hace_roundtrip(tmp_path: Path):
    pipe = _pipe_entrenado()
    ruta = tmp_path / "recompra_v1"
    guardar(pipe, ruta, _metadatos_validos())
    pipe_cargado, meta = cargar(ruta)
    # Mismas predicciones.
    rng = np.random.default_rng(7)
    X_test = pd.DataFrame(
        {
            "monetario": rng.uniform(10, 1000, 10),
            "frecuencia": rng.integers(1, 10, 10),
            "productos_distintos": rng.integers(1, 50, 10),
            "recencia": rng.uniform(0, 500, 10),
            "antiguedad": rng.uniform(0, 800, 10),
            "tasa_devolucion": rng.uniform(0, 0.3, 10),
            "pais": rng.choice(["UK", "otro"], 10),
        }
    )
    np.testing.assert_array_equal(pipe.predict(X_test), pipe_cargado.predict(X_test))
    assert meta["autor"] == "David"


def test_metadatos_json_versionado_con_sklearn_autocompletado(tmp_path: Path):
    pipe = _pipe_entrenado()
    ruta = tmp_path / "recompra_v1"
    guardar(pipe, ruta, _metadatos_validos())
    meta = json.loads((ruta / "metadatos.json").read_text())
    assert "sklearn" in meta["versiones"]
    assert meta["versiones"]["sklearn"]  # no vacio


def test_falta_un_campo_obligatorio_lanza_metadatosincompletos(tmp_path: Path):
    pipe = _pipe_entrenado()
    meta_malos = _metadatos_validos()
    del meta_malos["umbral_acordado"]
    with pytest.raises(MetadatosIncompletos, match="umbral_acordado"):
        guardar(pipe, tmp_path / "recompra_v1", meta_malos)


def test_cargar_falla_con_version_sklearn_distinta(tmp_path: Path, monkeypatch):
    pipe = _pipe_entrenado()
    ruta = tmp_path / "recompra_v1"
    meta = _metadatos_validos()
    meta["versiones"]["sklearn"] = "0.1.0"  # version ficticia
    guardar(pipe, ruta, meta)
    with pytest.raises(VersionIncompatible, match="scikit-learn"):
        cargar(ruta)


def test_cargar_falla_con_mensaje_claro_si_no_existe(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadatos.json"):
        cargar(tmp_path / "no_existe")

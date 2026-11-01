"""Fixtures de los tests del nucleo de prediccion (Unidad 4).

`artefacto_chico` entrena un pipeline pequeno con el preprocesador real y lo
guarda con `registro.guardar()` y los metadatos de la Tabla 7 del OVA U3 mas
los campos de la U4. Asi los tests corren en CI sin datos ni artefacto real.
`artefacto_real` apunta a `models/recompra_v1/` y se omite si no existe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from segmentacion.models.registro import guardar
from segmentacion.pipelines.preprocesamiento import CAT, NUM_LOG, NUM_STD, construir_preprocesador
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTEFACTO_REAL = PROJECT_ROOT / "models" / "recompra_v1"
VARIABLES = NUM_LOG + NUM_STD + CAT

CASO = {
    "monetario": 2350.0,
    "frecuencia": 9,
    "productos_distintos": 60,
    "recencia": 12,
    "antiguedad": 400,
    "tasa_devolucion": 0.0,
    "pais": "United Kingdom",
}


def _metadatos_chicos() -> dict:
    return {
        "fecha_corte": "2011-06-01",
        "horizonte": 90,
        "variables_en_orden": VARIABLES,
        "semilla": 42,
        "versiones": {"python": "3.11", "polars": "2.0"},
        "datos": {"ruta": "sintetico", "hash": "0000"},
        "auc_validacion": 0.5,
        "umbral_acordado": 0.3,
        "fecha": "2026-10-26",
        "autor": "tests",
        "version_modelo": "recompra_test",
        "tabla_umbrales": [
            {"umbral": u, "n_senialados": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            for u in (0.3, 0.4, 0.5, 0.6)
        ],
    }


@pytest.fixture(scope="session")
def artefacto_chico(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame(
        {
            "monetario": rng.uniform(10, 5000, n),
            "frecuencia": rng.integers(1, 20, n),
            "productos_distintos": rng.integers(1, 100, n),
            "recencia": rng.uniform(0, 500, n),
            "antiguedad": rng.uniform(0, 800, n),
            "tasa_devolucion": rng.uniform(0, 0.3, n),
            "pais": rng.choice(["UK", "otro"], n),
        }
    )[VARIABLES]
    y = (X["recencia"] < X["recencia"].median()).astype(int)
    pipe = Pipeline(
        [("pre", construir_preprocesador()), ("mod", LogisticRegression(max_iter=1000))]
    ).fit(X, y)
    ruta = tmp_path_factory.mktemp("modelo") / "recompra_test"
    guardar(pipe, ruta, _metadatos_chicos())
    return ruta


@pytest.fixture
def ruta_modelo(artefacto_chico: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hace que `RUTA_MODELO` apunte al artefacto chico para las interfaces."""
    monkeypatch.setenv("RUTA_MODELO", str(artefacto_chico))
    return artefacto_chico


@pytest.fixture(scope="session")
def artefacto_real() -> Path:
    if not (ARTEFACTO_REAL / "pipeline.joblib").exists():
        pytest.skip("models/recompra_v1/pipeline.joblib no existe; ejecuta scripts/entrenar_recompra.py")
    return ARTEFACTO_REAL


@pytest.fixture(scope="session")
def metadatos_versionados() -> dict:
    return json.loads((ARTEFACTO_REAL / "metadatos.json").read_text())

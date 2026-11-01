"""Las interfaces no tienen logica propia: responden lo mismo que el Predictor."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from segmentacion.servicio import Predictor

from .conftest import CASO, PROJECT_ROOT

PALABRAS_PROHIBIDAS = ("joblib", "predict_proba", "segmentacion.models", "segmentacion.pipelines")


def _cargar_api(ruta_modelo: Path):
    """Importa `api.main` con `RUTA_MODELO` apuntando al artefacto chico."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    sys.modules.pop("api.main", None)
    return importlib.import_module("api.main")


def test_fastapi_y_predictor_dan_lo_mismo(ruta_modelo: Path):
    main = _cargar_api(ruta_modelo)
    directo = Predictor.desde_carpeta(ruta_modelo).predecir(CASO)
    with TestClient(main.app) as cliente:
        assert cliente.get("/salud").json()["estado"] == "ok"
        via_api = cliente.post("/predecir", json=CASO).json()
    assert abs(via_api["probabilidad"] - directo["probabilidad"]) < 1e-6
    assert via_api["recompra"] == directo["recompra"]


def test_fastapi_rechaza_entrada_invalida(ruta_modelo: Path):
    main = _cargar_api(ruta_modelo)
    with TestClient(main.app) as cliente:
        respuesta = cliente.post("/predecir", json=CASO | {"recencia": -3})
    assert respuesta.status_code == 422


def test_ninguna_interfaz_tiene_logica_de_modelo():
    """Busqueda literal: las interfaces no cargan joblib ni llaman al modelo; solo al nucleo."""
    for archivo in ("app/recompra_streamlit.py", "api/main.py"):
        texto = (PROJECT_ROOT / archivo).read_text(encoding="utf-8")
        for palabra in PALABRAS_PROHIBIDAS:
            assert palabra not in texto, f"{archivo} contiene '{palabra}'"
        importes = [ln for ln in texto.splitlines() if ln.startswith("from segmentacion")]
        assert importes and all(ln.startswith("from segmentacion.servicio import") for ln in importes)

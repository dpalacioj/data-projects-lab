"""Tests de `segmentacion.evaluation.metricas`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from segmentacion.evaluation.metricas import evaluar_clasificacion, regla_recencia


def test_evaluar_clasificacion_devuelve_las_seis_metricas():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
    out = evaluar_clasificacion(y_true, y_proba, umbral=0.5)
    assert set(out.keys()) == {"auc", "ap", "accuracy", "precision", "recall", "f1"}
    assert out["accuracy"] == pytest.approx(4 / 5)  # 1 falso negativo @ 0.4


def test_evaluar_clasificacion_umbral_cambia_el_resultado():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
    a = evaluar_clasificacion(y_true, y_proba, umbral=0.3)
    b = evaluar_clasificacion(y_true, y_proba, umbral=0.6)
    assert a["recall"] > b["recall"]


def test_regla_recencia_predice_1_cuando_recencia_es_baja():
    X = pd.DataFrame({"recencia": [10, 50, 89, 90, 150]})
    pred = regla_recencia(X, dias=90)
    assert pred.tolist() == [1, 1, 1, 0, 0]

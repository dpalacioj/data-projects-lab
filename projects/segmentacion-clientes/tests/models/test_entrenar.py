"""Tests de `segmentacion.models.entrenar`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from segmentacion.models.entrenar import MODELOS, entrenar
from segmentacion.pipelines.preprocesamiento import construir_preprocesador
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


@pytest.fixture
def dataset_minimo():
    rng = np.random.default_rng(42)
    n = 60
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
    # Etiqueta ligeramente correlacionada con recencia para que AUC > 0.5.
    y = (X["recencia"] < X["recencia"].median()).astype(int).to_numpy()
    return X, y


def test_modelos_contiene_las_tres_familias_del_ova():
    assert set(MODELOS.keys()) == {
        "Regresion logistica",
        "Arbol de decision",
        "Bosque aleatorio",
    }


def test_entrenar_devuelve_las_cuatro_metricas_cv(dataset_minimo):
    X, y = dataset_minimo
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = Pipeline([("pre", construir_preprocesador()), ("mod", MODELOS["Regresion logistica"])])
    out = entrenar(pipe, X, y, cv=cv)
    assert set(out.keys()) == {"cv_auc", "cv_auc_sd", "cv_ap", "cv_f1"}
    assert 0.0 <= out["cv_auc"] <= 1.0
    assert 0.0 <= out["cv_ap"] <= 1.0
    assert 0.0 <= out["cv_f1"] <= 1.0
    assert out["cv_auc_sd"] >= 0.0

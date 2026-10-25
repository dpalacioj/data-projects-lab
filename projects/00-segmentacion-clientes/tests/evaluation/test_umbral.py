"""Tests de `segmentacion.evaluation.umbral.tabla_umbrales`."""

from __future__ import annotations

import numpy as np
import polars as pl
from segmentacion.evaluation.umbral import tabla_umbrales


def test_devuelve_un_df_con_columnas_esperadas():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
    out = tabla_umbrales(y_true, y_proba)
    assert out.columns == ["umbral", "n_senialados", "precision", "recall", "f1"]
    assert out.height == 4  # default: 0.3, 0.4, 0.5, 0.6


def test_umbral_mas_alto_senialan_menos():
    y_proba = np.linspace(0.0, 1.0, 20)
    y_true = (y_proba > 0.5).astype(int)
    out = tabla_umbrales(y_true, y_proba, umbrales=(0.3, 0.5, 0.7))
    n = out["n_senialados"].to_list()
    assert n[0] >= n[1] >= n[2]


def test_recall_cae_con_umbral_mas_alto_cuando_predicciones_son_bien_ordenadas():
    y_proba = np.linspace(0.0, 1.0, 50)
    y_true = (np.linspace(0, 1, 50) > 0.3).astype(int)
    out = tabla_umbrales(y_true, y_proba, umbrales=(0.3, 0.7))
    assert (
        out.filter(pl.col("umbral") == 0.3)["recall"].item()
        > out.filter(pl.col("umbral") == 0.7)["recall"].item()
    )

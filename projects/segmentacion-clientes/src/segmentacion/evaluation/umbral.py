"""Tabla de metricas por umbral para la Unidad 3.

Firma citada por `SPEC_procesamiento_U3.md` §4:

    tabla_umbrales(y, p, umbrales=(0.3, 0.4, 0.5, 0.6))
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score


def tabla_umbrales(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    umbrales: Iterable[float] = (0.3, 0.4, 0.5, 0.6),
) -> pl.DataFrame:
    """Devuelve `pl.DataFrame` con `umbral`, `n_senialados`, `precision`, `recall`, `f1`."""
    filas = []
    for t in umbrales:
        y_pred = (y_proba >= t).astype(int)
        filas.append(
            {
                "umbral": float(t),
                "n_senialados": int(y_pred.sum()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pl.DataFrame(filas)

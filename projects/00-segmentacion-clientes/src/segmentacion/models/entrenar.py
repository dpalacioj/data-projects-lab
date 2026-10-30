"""Modelos y utilidad de entrenamiento para la Unidad 3.

Firmas citadas por `SPEC_procesamiento_U3.md` §4:

    MODELOS = {...}  # tres modelos del curso
    entrenar(pipeline, X, y, cv) -> dict

Los nombres y los hiperparametros son los de la spec (seccion 6, Figura 8) y no
se cambian: regresion logistica, arbol de decision con profundidad 4, bosque
aleatorio con 300 arboles y hoja minima 5. La semilla es `config.SEMILLA`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from segmentacion.config import SEMILLA

MODELOS: dict[str, Any] = {
    "Regresión logística": LogisticRegression(max_iter=1000, random_state=SEMILLA),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=4, random_state=SEMILLA),
    "Bosque aleatorio": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=SEMILLA,
        n_jobs=-1,
    ),
}


def entrenar(
    pipeline: Pipeline,
    X: Any,
    y: np.ndarray,
    cv: BaseCrossValidator,
) -> dict[str, float]:
    """Entrena `pipeline` con validacion cruzada estratificada sobre `(X, y)`.

    Devuelve un dict con `cv_auc` (media), `cv_auc_sd` (desviacion),
    `cv_ap` (media de precision media) y `cv_f1` (media F1). No ajusta
    todavia sobre todo `X` — el que llama decide si hace `pipeline.fit(X, y)`
    como paso siguiente.
    """
    resultado = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "average_precision", "f1"],
        n_jobs=-1,
    )
    return {
        "cv_auc": float(np.mean(resultado["test_roc_auc"])),
        "cv_auc_sd": float(np.std(resultado["test_roc_auc"])),
        "cv_ap": float(np.mean(resultado["test_average_precision"])),
        "cv_f1": float(np.mean(resultado["test_f1"])),
    }

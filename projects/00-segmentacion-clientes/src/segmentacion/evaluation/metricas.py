"""Metricas y helpers de demostracion de fuga para la Unidad 3.

Firmas citadas por `SPEC_procesamiento_U3.md` §4:

    evaluar_clasificacion(y, p, umbral) -> dict
    regla_recencia(X, dias=90)

Mas los dos helpers del cuaderno 06 (fuga temporal) y el analisis de la curva
de profundidad del arbol (cuaderno 09).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from segmentacion.config import SEMILLA


def evaluar_clasificacion(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    umbral: float = 0.5,
) -> dict[str, float]:
    """Metricas al `umbral` dado: AUC, AP, accuracy, precision, recall y F1.

    `y_proba` es la probabilidad de la clase positiva; las decisiones duras
    se derivan comparando contra `umbral`.
    """
    y_pred = (y_proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "ap": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def regla_recencia(X: pd.DataFrame, dias: int = 90) -> np.ndarray:
    """Linea base: predice recompra = 1 si `recencia < dias`."""
    return (X["recencia"] < dias).astype(int).to_numpy()


def _pipe_logreg(preprocesador) -> Pipeline:
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        [
            ("pre", preprocesador),
            ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA)),
        ]
    )


def fuga_recencia(
    XA: pd.DataFrame,
    yA: np.ndarray,
    XB: pd.DataFrame,
    yB: np.ndarray,
    recencia_fuga_A: np.ndarray,
    recencia_fuga_B: np.ndarray,
    construir_preprocesador: Any,
) -> dict[str, float]:
    """Demostracion: usar `recencia_fuga` (datos posteriores al corte) infla AUC.

    Devuelve `{cv_auc, test_auc}` para una regresion logistica entrenada con
    la recencia contaminada. Comparar contra la version honesta en el cuaderno 06.
    """
    XA_l = XA.copy()
    XA_l["recencia"] = recencia_fuga_A
    XB_l = XB.copy()
    XB_l["recencia"] = recencia_fuga_B
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    pipe = _pipe_logreg(construir_preprocesador())
    cv_auc = cross_validate(pipe, XA_l, yA, cv=cv, scoring="roc_auc")["test_score"].mean()
    pipe.fit(XA_l, yA)
    test_auc = roc_auc_score(yB, pipe.predict_proba(XB_l)[:, 1])
    return {"cv_auc": float(cv_auc), "test_auc": float(test_auc)}


def fuga_escalado(
    XA: pd.DataFrame,
    XB: pd.DataFrame,
    columnas_num_std: list[str],
) -> dict[str, float]:
    """Compara `StandardScaler.fit(A ∪ B)` con `StandardScaler.fit(A)`.

    Devuelve la media de la primera columna de `columnas_num_std`
    (`recencia` por convencion). La spec lo reporta como 186,9 vs 168,7.
    """
    sc_todo = StandardScaler().fit(
        np.vstack([XA[columnas_num_std].to_numpy(), XB[columnas_num_std].to_numpy()])
    )
    sc_train = StandardScaler().fit(XA[columnas_num_std].to_numpy())
    return {
        "media_recencia_todo": float(sc_todo.mean_[0]),
        "media_recencia_train": float(sc_train.mean_[0]),
    }


def curva_profundidad(
    XA: pd.DataFrame,
    yA: np.ndarray,
    XB: pd.DataFrame,
    yB: np.ndarray,
    construir_preprocesador: Any,
    profundidades: list[int | None] = (1, 2, 3, 4, 6, 8, 12, None),
) -> list[tuple[int | None, float, float]]:
    """Curva de sobreajuste del arbol: AUC en A vs AUC en B para cada profundidad.

    Es la cifra del cuaderno 09 que muestra como la memorizacion sube AUC(A)
    pero baja AUC(B).
    """
    curva: list[tuple[int | None, float, float]] = []
    for d in profundidades:
        pipe = Pipeline(
            [
                ("pre", construir_preprocesador()),
                ("modelo", DecisionTreeClassifier(max_depth=d, random_state=SEMILLA)),
            ]
        ).fit(XA, yA)
        auc_A = roc_auc_score(yA, pipe.predict_proba(XA)[:, 1])
        auc_B = roc_auc_score(yB, pipe.predict_proba(XB)[:, 1])
        curva.append((d, float(auc_A), float(auc_B)))
    return curva

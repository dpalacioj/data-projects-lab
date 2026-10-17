"""Metricas, umbrales y demostraciones de fuga para la Unidad 3."""

from segmentacion.evaluation.metricas import (
    curva_profundidad,
    evaluar_clasificacion,
    fuga_escalado,
    fuga_recencia,
    regla_recencia,
)
from segmentacion.evaluation.umbral import tabla_umbrales

__all__ = [
    "evaluar_clasificacion",
    "regla_recencia",
    "fuga_recencia",
    "fuga_escalado",
    "curva_profundidad",
    "tabla_umbrales",
]

"""Pipelines de scikit-learn para la Unidad 3."""

from segmentacion.pipelines.preprocesamiento import (
    CAT,
    NUM_LOG,
    NUM_STD,
    construir_preprocesador,
)

__all__ = ["CAT", "NUM_LOG", "NUM_STD", "construir_preprocesador"]

"""Modelos: clustering (U2), caso supervisado y registro de artefactos (U3)."""

from segmentacion.models.clustering import (
    evaluate_clustering,
    find_optimal_k,
    fit_dbscan,
    fit_kmeans,
)
from segmentacion.models.entrenar import MODELOS, entrenar
from segmentacion.models.profiling import build_segment_profiles, label_segments
from segmentacion.models.registro import (
    MetadatosIncompletos,
    VersionIncompatible,
    cargar,
    guardar,
)

__all__ = [
    "evaluate_clustering",
    "find_optimal_k",
    "fit_dbscan",
    "fit_kmeans",
    "build_segment_profiles",
    "label_segments",
    "MODELOS",
    "entrenar",
    "guardar",
    "cargar",
    "MetadatosIncompletos",
    "VersionIncompatible",
]

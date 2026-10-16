"""Construccion de features: RFM en Polars + pipeline sklearn (legacy)."""
from segmentacion.features.preprocessing import build_preprocessing_pipeline
from segmentacion.features.rfm import construir_rfm

__all__ = ["construir_rfm", "build_preprocessing_pipeline"]

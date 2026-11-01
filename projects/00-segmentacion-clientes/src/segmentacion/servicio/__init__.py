"""Nucleo de prediccion de la Unidad 4.

Es lo unico que las interfaces (Streamlit, FastAPI) necesitan
importar. Ninguna interfaz carga el artefacto, ordena columnas ni aplica el
umbral por su cuenta: todo pasa por `Predictor`.
"""

from segmentacion.servicio.entrada import EntradaCliente
from segmentacion.servicio.prediccion import Predictor

__all__ = ["EntradaCliente", "Predictor"]

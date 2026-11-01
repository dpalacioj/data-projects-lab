"""Predictor: carga el artefacto una vez, valida la entrada y predice.

No sabe nada de pantallas ni de rutas web. Lee del `metadatos.json` de la
Unidad 3 todo lo que necesita: `variables_en_orden`, `umbral_acordado`,
`fecha_corte` y `versiones`. La ruta del artefacto viene de
`segmentacion.config.RUTA_MODELO` (o de la variable de entorno homonima).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from segmentacion import config
from segmentacion.models.registro import cargar
from segmentacion.servicio.entrada import EntradaCliente


class Predictor:
    """Envuelve el pipeline serializado y sus metadatos."""

    def __init__(self, pipeline: Any, metadatos: dict[str, Any], version_modelo: str) -> None:
        self.pipeline = pipeline
        self.metadatos = metadatos
        self.variables: list[str] = list(metadatos["variables_en_orden"])
        self.umbral_acordado: float = float(metadatos["umbral_acordado"])
        self.version_modelo = version_modelo

    @classmethod
    def desde_carpeta(cls, ruta: Path | str | None = None) -> Predictor:
        """Carga `pipeline.joblib` + `metadatos.json` con `registro.cargar()` (Unidad 3).

        Sin argumento usa `RUTA_MODELO`: la variable de entorno si existe, o el
        valor por defecto de `segmentacion.config`.
        """
        ruta = Path(ruta or os.getenv("RUTA_MODELO", config.RUTA_MODELO))
        pipeline, metadatos = cargar(ruta)
        version = metadatos.get("version_modelo", ruta.name)
        return cls(pipeline, metadatos, version)

    def _umbral(self, umbral: float | None) -> float:
        return self.umbral_acordado if umbral is None else float(umbral)

    def predecir(self, entrada: EntradaCliente | dict, umbral: float | None = None) -> dict:
        """Probabilidad de recompra de UN cliente y la decision con el umbral."""
        if not isinstance(entrada, EntradaCliente):
            entrada = EntradaCliente(**entrada)
        fila = pd.DataFrame([entrada.model_dump()])[self.variables]
        prob = float(self.pipeline.predict_proba(fila)[0, 1])
        u = self._umbral(umbral)
        return {
            "probabilidad": round(prob, 4),
            "recompra": bool(prob >= u),
            "umbral": u,
            "version_modelo": self.version_modelo,
        }

    def predecir_lote(self, df: pl.DataFrame, umbral: float | None = None) -> pl.DataFrame:
        """Misma logica para VARIOS clientes; devuelve el DataFrame ordenado por probabilidad."""
        filas = [
            EntradaCliente(**r).model_dump() for r in df.select(self.variables).iter_rows(named=True)
        ]
        prob = self.pipeline.predict_proba(pd.DataFrame(filas)[self.variables])[:, 1]
        u = self._umbral(umbral)
        return df.with_columns(
            pl.Series("probabilidad", prob).round(4),
            pl.Series("recompra", prob >= u),
        ).sort("probabilidad", descending=True)

    def salud(self) -> dict:
        """Que modelo responde y con que datos y versiones se construyo."""
        m = self.metadatos
        return {
            "estado": "ok",
            "version_modelo": self.version_modelo,
            "fecha_corte": m["fecha_corte"],
            "sklearn": m["versiones"]["sklearn"],
        }

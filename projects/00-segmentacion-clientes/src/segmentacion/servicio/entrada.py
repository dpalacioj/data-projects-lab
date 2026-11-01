"""Contrato de entrada de un cliente (Unidad 4).

Los campos siguen, en nombre y orden, `metadatos.json["variables_en_orden"]`
del artefacto `models/recompra_v1/` (= `NUM_LOG + NUM_STD + CAT` del
preprocesador de la Unidad 3). `tests/servicio/test_entrada.py` verifica que
coinciden.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

PAISES_UK = {"uk", "united kingdom", "reino unido"}


class EntradaCliente(BaseModel):
    """Variables de un cliente calculadas antes de la fecha de corte."""

    monetario: float = Field(ge=0, description="Gasto total antes del corte (£)")
    frecuencia: int = Field(ge=1, description="Numero de facturas antes del corte")
    productos_distintos: int = Field(ge=1, description="Productos distintos comprados")
    recencia: float = Field(ge=0, description="Dias desde la ultima compra")
    antiguedad: float = Field(ge=0, description="Dias desde la primera compra")
    tasa_devolucion: float = Field(ge=0, description="Facturas canceladas / frecuencia")
    pais: str = Field(description="Pais del cliente; se normaliza a UK u otro")

    @field_validator("pais")
    @classmethod
    def normalizar_pais(cls, valor: str) -> str:
        """`UK`, `United Kingdom` y `Reino Unido` -> `UK`; cualquier otro -> `otro`."""
        return "UK" if valor.strip().lower() in PAISES_UK else "otro"

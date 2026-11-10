"""Modelos Pydantic que describen la forma esperada del JSON de entrada.

Un archivo JSON diario trae una lista de pedidos. Cada pedido tiene:

    {
        "pedido_id":   "P-2024-00042",
        "fecha":       "2024-01-15T10:32:00",
        "canal":       "web" | "app" | "marketplace",
        "cliente": {
            "cliente_id": "C-01823",
            "nombre":     "...",
            "email":      "...",
            "ciudad":     "...",
            "pais":       "..."
        },
        "items": [
            {
                "producto_id":    "SKU-00123",
                "nombre":         "...",
                "categoria":      "...",
                "cantidad":       2,
                "precio_unitario": 19.90
            },
            ...
        ]
    }

Pydantic nos da validacion: si el JSON viene mal formado, falla aqui con un
mensaje claro, no 3 pasos despues con un `KeyError` incomprensible.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClienteJSON(BaseModel):
    cliente_id: str = Field(..., min_length=1)
    nombre: str
    email: str  # no uso EmailStr para no pedir dependencia extra; validamos abajo
    ciudad: str
    pais: str

    @field_validator("email")
    @classmethod
    def _email_tiene_arroba(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"email invalido: {v!r}")
        return v


class ItemJSON(BaseModel):
    producto_id: str = Field(..., min_length=1)
    nombre: str
    categoria: str
    cantidad: int = Field(..., gt=0)
    precio_unitario: float = Field(..., ge=0)


class PedidoJSON(BaseModel):
    pedido_id: str = Field(..., min_length=1)
    fecha: datetime
    canal: Literal["web", "app", "marketplace"]
    cliente: ClienteJSON
    items: list[ItemJSON] = Field(..., min_length=1)


class LoteJSON(BaseModel):
    """Un archivo JSON diario: metadatos + lista de pedidos."""
    fecha_lote: str
    version_esquema: str = "1.0"
    pedidos: list[PedidoJSON]

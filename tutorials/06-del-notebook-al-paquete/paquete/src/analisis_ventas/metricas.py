"""Metricas del analisis de ventas."""

from __future__ import annotations

import pandas as pd


def precio_promedio_ponderado(df: pd.DataFrame) -> float:
    """Precio promedio ponderado por cantidad.

    Suma(Price * Quantity) / Suma(Quantity). Es la medida honesta de "cuanto
    cobramos en promedio por unidad", en vez de un promedio simple de Price
    que ignora que algunas filas venden 20 unidades y otras 1.
    """
    ingresos = (df["Price"] * df["Quantity"]).sum()
    unidades = df["Quantity"].sum()
    return float(ingresos / unidades)

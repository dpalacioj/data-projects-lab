"""Generador deterministico del dataset del tutorial.

5000 filas de ventas de un retailer online de regalos, con la misma forma
de columnas que Online Retail II (caso guia del repo), pero sin descarga
externa. Incluye cancelaciones (facturas con prefijo "C") y algunos precios
fuera de rango, para que las reglas de limpieza tengan algo que hacer.

Uso:
    import pandas as pd
    from datos_sinteticos import generar_datos
    df = generar_datos()
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEMILLA_POR_DEFECTO: int = 42
N_FILAS: int = 5000


def generar_datos(semilla: int = SEMILLA_POR_DEFECTO) -> pd.DataFrame:
    """Genera un DataFrame determinista de ventas sinteticas."""
    rng = np.random.default_rng(semilla)

    productos = rng.choice(
        ["A", "B", "C", "D", "E", "F"],
        size=N_FILAS,
        p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03],
    )
    # Precios ~ lognormal centrado en 10 unidades monetarias; ~2% caen fuera
    # de rango valido (negativos o > 100), para ejercitar la regla de precio.
    precios_base = rng.lognormal(mean=2.0, sigma=0.6, size=N_FILAS).round(2)
    outliers = rng.random(N_FILAS) < 0.02
    precios = np.where(outliers, rng.choice([-1.0, 500.0], size=N_FILAS), precios_base)

    cantidades = rng.integers(1, 15, size=N_FILAS)

    # 10% de las filas son cancelaciones (invoice con prefijo "C"), con
    # cantidad negativa.
    es_cancelacion = rng.random(N_FILAS) < 0.10
    numeros_factura = rng.integers(10000, 99999, size=N_FILAS)
    invoices = np.where(
        es_cancelacion,
        np.array([f"C{n}" for n in numeros_factura]),
        np.array([f"A{n}" for n in numeros_factura]),
    )
    cantidades = np.where(es_cancelacion, -cantidades, cantidades)

    return pd.DataFrame({
        "Invoice":  invoices,
        "Product":  productos,
        "Price":    precios,
        "Quantity": cantidades,
    })

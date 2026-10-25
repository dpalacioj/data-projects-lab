"""Verifica que el bosque entrenado dos veces produce las mismas predicciones.

La reproducibilidad depende de `.sort("Customer ID")` dentro de
`features.cohortes.cohorte`: `polars.unique()` no conserva el orden y, sin
ese sort final, un re-entrenamiento del bosque cambia las particiones de los
`StratifiedKFold` internos y los nodos que elige cada arbol. Sin el sort,
las predicciones difieren silenciosamente.

Esta prueba se corre con filas inventadas (sin tocar disco). Para la version
sobre el parquet real, ver `test_cifras_recompra::test_reproducibilidad_del_bosque`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from segmentacion.config import SEMILLA
from segmentacion.features.cohortes import cohorte
from segmentacion.pipelines.preprocesamiento import CAT, NUM_LOG, NUM_STD, construir_preprocesador
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def _ventas_inventadas() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Genera un dataset reproducible con 100 clientes y actividad en dos años."""
    rng = np.random.default_rng(42)
    base = datetime(2010, 1, 1)
    filas = []
    cancelaciones = []
    for cid in range(1_000, 1_100):
        n_compras = int(rng.integers(2, 15))
        for i in range(n_compras):
            dias = int(rng.integers(0, 700))
            filas.append(
                {
                    "Invoice": f"A{cid}-{i}",
                    "StockCode": f"X{rng.integers(1, 50)}",
                    "Description": "x",
                    "Quantity": int(rng.integers(1, 20)),
                    "InvoiceDate": base + timedelta(days=dias),
                    "Price": float(rng.uniform(1, 50)),
                    "Customer ID": cid,
                    "Country": "United Kingdom" if rng.random() > 0.1 else "France",
                }
            )
        if rng.random() < 0.3:
            cancelaciones.append(
                {
                    "Invoice": f"C{cid}",
                    "StockCode": "X",
                    "Description": "x",
                    "Quantity": -1,
                    "InvoiceDate": base + timedelta(days=int(rng.integers(0, 500))),
                    "Price": 1.0,
                    "Customer ID": cid,
                    "Country": "United Kingdom",
                }
            )
    ventas = pl.DataFrame(filas).with_columns(
        (pl.col("Quantity") * pl.col("Price")).alias("Importe")
    )
    canc = pl.DataFrame(cancelaciones) if cancelaciones else pl.DataFrame(schema=ventas.schema)
    return ventas, canc


def test_bosque_dos_veces_produce_predicciones_identicas():
    ventas, canc = _ventas_inventadas()
    corte = datetime(2011, 6, 1)
    coh = cohorte(ventas, canc, corte)
    cols = NUM_LOG + NUM_STD + CAT
    X = coh.select(cols).to_pandas()
    y = coh["recompra"].to_numpy()

    def entrenar_y_predecir():
        # n_jobs=1 para que la agregacion de las probabilidades de los 300
        # arboles ocurra en orden determinista. El bosque de produccion
        # (MODELOS["Bosque aleatorio"]) usa n_jobs=-1 por velocidad; los
        # arboles son identicos pero la suma paralela introduce ruido a
        # nivel epsilon (~2e-16) que rompe la igualdad exacta. Esta prueba
        # verifica que el orden de las filas (via .sort("Customer ID") en
        # cohorte) es suficiente para la reproducibilidad logica.
        bosque = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEMILLA, n_jobs=1
        )
        pipe = Pipeline([("pre", construir_preprocesador()), ("mod", bosque)]).fit(X, y)
        return pipe.predict_proba(X)[:, 1]

    p1 = entrenar_y_predecir()
    p2 = entrenar_y_predecir()
    np.testing.assert_array_equal(p1, p2)

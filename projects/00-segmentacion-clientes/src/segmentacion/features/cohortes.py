"""Cohorte supervisada de clientes para la Unidad 3.

Firma publica:

    cohorte(ventas, cancelaciones, corte, horizonte=90) -> pl.DataFrame

Compone:

- RFM extendido con `features.rfm.construir_rfm` (recencia, antiguedad,
  frecuencia, monetario, productos_distintos).
- `pais ∈ {UK, otro}` a partir del primer `Country` observado antes del corte.
- `tasa_devolucion` con `features.devoluciones.tasa_devolucion`.
- Etiqueta `recompra` con `features.etiquetas.etiqueta_recompra`.
- `recencia_fuga`: feature de demostracion que incluye datos POSTERIORES al
  corte; sirve para el cuaderno 06 (demostracion de fuga temporal). No se usa
  en los modelos finales.

El DataFrame termina con `.sort("Customer ID")` porque `Polars.unique()` no
conserva el orden; sin esa linea el bosque entrenado dos veces produce
predicciones distintas, rompiendo la reproducibilidad (ver
`tests/test_reproducibilidad.py`).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from segmentacion.features.devoluciones import tasa_devolucion
from segmentacion.features.etiquetas import etiqueta_recompra
from segmentacion.features.rfm import construir_rfm

Frame = pl.DataFrame | pl.LazyFrame


def cohorte(
    ventas: Frame,
    cancelaciones: Frame,
    corte: datetime,
    horizonte: int = 90,
) -> pl.DataFrame:
    """Construye la cohorte de clientes a `corte` con etiqueta a `+ horizonte` dias."""
    v = ventas.lazy() if isinstance(ventas, pl.DataFrame) else ventas

    rfm = construir_rfm(v, corte)
    hist = v.filter(pl.col("InvoiceDate") < corte)
    pais = hist.group_by("Customer ID").agg(pl.col("Country").first().alias("pais")).collect()
    feats = rfm.join(pais, on="Customer ID", how="left").with_columns(
        pl.when(pl.col("pais") == "United Kingdom")
        .then(pl.lit("UK"))
        .otherwise(pl.lit("otro"))
        .alias("pais")
    )

    tasa = tasa_devolucion(
        cancelaciones,
        feats.select("Customer ID", "frecuencia"),
        corte,
    )
    feats = feats.join(tasa, on="Customer ID", how="left")

    y = etiqueta_recompra(v, corte, horizonte_dias=horizonte)
    feats = feats.join(y, on="Customer ID", how="left").with_columns(
        pl.col("recompra").fill_null(0).cast(pl.Int64)
    )

    # Feature de demostracion de fuga: recencia calculada con TODAS las compras
    # (incluye el futuro). Es la que el cuaderno 06 muestra como contraste.
    fuga = (
        v.group_by("Customer ID")
        .agg((pl.lit(corte) - pl.col("InvoiceDate").max()).dt.total_days().alias("recencia_fuga"))
        .collect()
    )
    feats = feats.join(fuga, on="Customer ID", how="left")

    return feats.sort("Customer ID")

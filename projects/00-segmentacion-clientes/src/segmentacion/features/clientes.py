"""Features extendidas por cliente para el flujo no supervisado (U2).

Firma publica:

    construir_features_extendidas(ventas, fecha_corte=None) -> pl.DataFrame

Produce la tabla que la app de segmentacion (`app/streamlit_app.py`) y el
cuaderno `10_segmentacion_no_supervisada.ipynb` consumen. Reutiliza
`features.rfm.construir_rfm` para R, F, M y variedad de productos, y agrega
tres variables de contexto que el clustering necesita:

- `AvgTicket`: ticket promedio por factura (monetario / frecuencia).
- `AvgQuantity`: cantidad promedio de unidades por factura.
- `CountryGroup`: pais reducido a `UK` u `otro`, igual que en U3.

Nombres de columnas en ingles por compatibilidad con `config.py`
(`EXTENDED_NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`), que a su vez son las
que la app y los modulos de clustering ya referencian. Si no se pasa
`fecha_corte` se usa el snapshot de la Unidad 2: `max(InvoiceDate) + 1 dia`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from segmentacion.features.rfm import construir_rfm

Frame = pl.DataFrame | pl.LazyFrame

_UK_LABELS = {"united kingdom", "uk", "reino unido"}


def _snapshot_por_defecto(ventas: Frame) -> datetime:
    v = ventas.lazy() if isinstance(ventas, pl.DataFrame) else ventas
    max_fecha = v.select(pl.col("InvoiceDate").max()).collect().item()
    return max_fecha + timedelta(days=1)


def construir_features_extendidas(
    ventas: Frame,
    fecha_corte: datetime | None = None,
) -> pl.DataFrame:
    """Construye la tabla de features extendidas por cliente."""
    if fecha_corte is None:
        fecha_corte = _snapshot_por_defecto(ventas)

    v = ventas.lazy() if isinstance(ventas, pl.DataFrame) else ventas
    v = v.filter(pl.col("InvoiceDate") < fecha_corte)

    rfm_basico = construir_rfm(v, fecha_corte)

    extras = (
        v.group_by("Customer ID")
        .agg(
            pl.col("Quantity").sum().alias("_cantidad_total"),
            pl.col("Country").mode().first().alias("_pais_raw"),
        )
        .collect()
    )

    return (
        rfm_basico.join(extras, on="Customer ID", how="inner")
        .with_columns(
            (pl.col("monetario") / pl.col("frecuencia")).alias("AvgTicket"),
            (pl.col("_cantidad_total") / pl.col("frecuencia")).alias("AvgQuantity"),
            pl.when(pl.col("_pais_raw").str.to_lowercase().is_in(list(_UK_LABELS)))
            .then(pl.lit("UK"))
            .otherwise(pl.lit("otro"))
            .alias("CountryGroup"),
        )
        .rename(
            {
                "recencia": "Recency",
                "frecuencia": "Frequency",
                "monetario": "Monetary",
                "productos_distintos": "ProductDiversity",
            }
        )
        .select(
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "AvgTicket",
            "ProductDiversity",
            "AvgQuantity",
            "CountryGroup",
        )
        .sort("Customer ID")
    )

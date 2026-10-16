"""
Construcción de features a nivel de cliente.

El análisis RFM (Recency, Frequency, Monetary) es una técnica clásica
de marketing que resume el comportamiento de un cliente en tres dimensiones:

- **Recency**:   cuán reciente fue su última compra (días).
- **Frequency**: cuántas compras distintas ha realizado.
- **Monetary**:  cuánto dinero ha gastado en total.

Aquí extendemos el RFM básico con tres métricas adicionales que ayudan
al clustering a distinguir comportamientos finos:

- **AvgTicket**:        gasto promedio por factura.
- **ProductDiversity**: cantidad de productos distintos comprados.
- **AvgQuantity**:      cantidad promedio de unidades por factura.
"""

from __future__ import annotations

import pandas as pd

from segmentacion.utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas de limpieza estándar para Online Retail II.

    Reglas:
    1. Eliminar filas sin ``CustomerID`` (no se pueden atribuir a un cliente).
    2. Eliminar cancelaciones (Invoice empieza con 'C').
    3. Eliminar filas con ``Quantity <= 0`` o ``Price <= 0``.
    4. Eliminar duplicados exactos.
    5. Calcular columna derivada ``LineTotal = Quantity * Price``.

    Parameters
    ----------
    df : pd.DataFrame
        Transacciones crudas tal como las devuelve ``load_raw_transactions``.

    Returns
    -------
    pd.DataFrame
        Transacciones limpias listas para feature engineering.
    """
    n_inicial = len(df)
    logger.info(f"Limpieza: filas iniciales = {n_inicial:,}")

    df = df.dropna(subset=["CustomerID"]).copy()
    logger.info(f"  - Sin CustomerID eliminadas: quedan {len(df):,}")

    # Las facturas canceladas en este dataset llevan prefijo 'C'.
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]
    logger.info(f"  - Sin cancelaciones: quedan {len(df):,}")

    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    logger.info(f"  - Cantidades/precios positivos: quedan {len(df):,}")

    df = df.drop_duplicates()
    logger.info(f"  - Sin duplicados: quedan {len(df):,}")

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]

    logger.info(
        f"Limpieza terminada: {len(df):,} filas "
        f"({100 * len(df) / n_inicial:.1f}% del original)"
    )
    return df


def build_rfm(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Calcula las tres métricas RFM clásicas a nivel de cliente.

    La fecha de referencia (``snapshot_date``) representa el "hoy" del análisis:
    típicamente la fecha de la última transacción del dataset + 1 día,
    para que ningún cliente tenga ``Recency = 0``.

    Parameters
    ----------
    df : pd.DataFrame
        Transacciones limpias (output de :func:`clean_transactions`).
    snapshot_date : pd.Timestamp, optional
        Fecha de referencia para calcular Recency. Si es ``None``,
        se usa la fecha máxima del dataset + 1 día.

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por ``CustomerID`` con columnas
        ``Recency``, ``Frequency`` y ``Monetary``.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    logger.info(f"Snapshot date para Recency: {snapshot_date.date()}")

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
    )

    logger.info(f"RFM calculado para {len(rfm):,} clientes")
    return rfm


def build_customer_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Construye el set extendido de features por cliente (RFM + comportamiento).

    Además del RFM clásico, agrega:

    - ``AvgTicket = Monetary / Frequency``
    - ``ProductDiversity``: nº de StockCode únicos comprados.
    - ``AvgQuantity``:      cantidad promedio de unidades por factura.

    Parameters
    ----------
    df : pd.DataFrame
        Transacciones limpias.
    snapshot_date : pd.Timestamp, optional
        Fecha de referencia para Recency.

    Returns
    -------
    pd.DataFrame
        Features por cliente, indexadas por ``CustomerID``.
    """
    rfm = build_rfm(df, snapshot_date=snapshot_date)

    extra = df.groupby("CustomerID").agg(
        ProductDiversity=("StockCode", "nunique"),
        AvgQuantity=("Quantity", "mean"),
    )

    features = rfm.join(extra)
    features["AvgTicket"] = features["Monetary"] / features["Frequency"]

    # Reordenar columnas para que el RFM clásico vaya primero.
    cols = ["Recency", "Frequency", "Monetary", "AvgTicket",
            "ProductDiversity", "AvgQuantity"]
    features = features[cols]

    logger.info(f"Features extendidas: {features.shape}")
    return features

"""
Interpretación de segmentos: traducir clusters en perfiles de negocio.

El objetivo de este módulo es convertir las etiquetas numéricas que
producen K-Means o DBSCAN en perfiles legibles para un equipo de
marketing, junto con recomendaciones de acción.

La heurística de etiquetado es deliberadamente simple y transparente:
se basa en cuartiles relativos de Recency, Frequency y Monetary, que
es como se hace tradicionalmente el "RFM scoring" en CRM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from segmentacion.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_segment_profiles(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Construye una tabla resumen con el perfil promedio de cada segmento.

    Devuelve, por cluster:

    - ``n_customers``: número de clientes asignados.
    - Promedio de cada feature (Recency, Frequency, Monetary, AvgTicket, ...).
    - ``share``: proporción del total.

    Parameters
    ----------
    features : pd.DataFrame
        Features por cliente (no escaladas).
    labels : np.ndarray
        Etiquetas de cluster, mismo orden que ``features``.

    Returns
    -------
    pd.DataFrame
        Indexado por cluster (incluyendo -1 si DBSCAN marcó ruido).
    """
    df = features.copy()
    df["cluster"] = labels

    summary = df.groupby("cluster").agg(["mean"])
    summary.columns = [c[0] for c in summary.columns]  # aplanar multi-index

    counts = df.groupby("cluster").size().rename("n_customers")
    summary.insert(0, "n_customers", counts)
    summary["share"] = summary["n_customers"] / summary["n_customers"].sum()

    return summary.round(2)


def label_segments(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un nombre de negocio + acción recomendada a cada cluster.

    La heurística usa las posiciones relativas (cuartiles) de Recency,
    Frequency y Monetary. Es interpretable y robusta a la escala.

    Reglas resumidas:
    - **Champions**:           Recency baja + Frequency alta + Monetary alto.
    - **Loyal**:               Frequency alta, Monetary medio/alto.
    - **At Risk**:             Recency alta + Monetary alto en el pasado.
    - **Hibernating**:         Recency alta + Frequency baja + Monetary bajo.
    - **New / One-time**:      Recency baja + Frequency baja.
    - **Outliers (DBSCAN)**:   etiqueta ``-1`` → "Atípicos".
    - **Otros**:               cualquier mezcla intermedia.

    Parameters
    ----------
    profiles : pd.DataFrame
        Output de :func:`build_segment_profiles`.

    Returns
    -------
    pd.DataFrame
        ``profiles`` con dos columnas extra: ``segment_name`` y ``action``.
    """
    profiles = profiles.copy()
    valid = profiles.index != -1

    # Cuartiles globales (entre los clusters, no entre clientes — los clusters
    # son pocos y queremos comparar segmentos entre sí).
    if valid.sum() > 0:
        r_med = profiles.loc[valid, "Recency"].median()
        f_med = profiles.loc[valid, "Frequency"].median()
        m_med = profiles.loc[valid, "Monetary"].median()
    else:
        r_med = f_med = m_med = 0

    names, actions = [], []
    for cluster_id, row in profiles.iterrows():
        if cluster_id == -1:
            names.append("Atípicos / Outliers")
            actions.append(
                "Revisar manualmente: pueden ser fraudes, mayoristas, "
                "o clientes con comportamiento muy distinto."
            )
            continue

        recent = row["Recency"] <= r_med
        frequent = row["Frequency"] >= f_med
        valuable = row["Monetary"] >= m_med

        if recent and frequent and valuable:
            names.append("Champions")
            actions.append(
                "Programa de fidelización VIP. Acceso anticipado a "
                "lanzamientos y recompensas exclusivas."
            )
        elif frequent and valuable:
            names.append("Clientes Leales")
            actions.append("Cross-sell de productos complementarios y campañas de referidos.")
        elif recent and not frequent:
            names.append("Clientes Nuevos / Ocasionales")
            actions.append(
                "Campaña de bienvenida y nurturing para incrementar frecuencia de compra."
            )
        elif not recent and valuable:
            names.append("En Riesgo / Hibernando")
            actions.append(
                "Campaña de reactivación con descuentos personalizados "
                "antes de perderlos definitivamente."
            )
        elif not recent and not frequent and not valuable:
            names.append("Inactivos / Perdidos")
            actions.append(
                "Última oportunidad: oferta agresiva. Si no responden, "
                "reducir presupuesto de marketing en este segmento."
            )
        else:
            names.append("Segmento Mixto")
            actions.append(
                "Analizar manualmente: combinación intermedia. Considerar A/B testing de mensajes."
            )

    profiles["segment_name"] = names
    profiles["action"] = actions
    return profiles

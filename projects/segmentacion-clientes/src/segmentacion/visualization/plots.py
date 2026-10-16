"""
Funciones reutilizables de visualización con matplotlib y plotly.

Cubre las gráficas más usadas en el proyecto:

- Distribuciones de variables (histograma + boxplot).
- Curvas de codo y silueta para K-Means.
- Curva k-distance para elegir ``eps`` de DBSCAN.
- Scatter 2D de clusters tras PCA.
- Radar / spider para perfilamiento de segmentos.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def plot_distributions(df: pd.DataFrame, columns: Iterable[str], bins: int = 40):
    """
    Histograma + boxplot lado a lado para cada columna numérica.

    Útil para detectar de un vistazo: sesgo, escala, outliers.
    """
    cols = list(columns)
    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(cols):
        axes[i, 0].hist(df[col].dropna(), bins=bins, edgecolor="black", alpha=0.75)
        axes[i, 0].set_title(f"Histograma · {col}")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("Frecuencia")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].boxplot(df[col].dropna(), vert=False)
        axes[i, 1].set_title(f"Boxplot · {col}")
        axes[i, 1].set_xlabel(col)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_elbow_and_silhouette(metrics: pd.DataFrame):
    """
    Curva del codo (inertia) y silueta lado a lado.

    Espera un DataFrame con índice ``k`` y columnas
    ``inertia`` y ``silhouette`` (output de ``find_optimal_k``).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(metrics.index, metrics["inertia"], "o-", color="steelblue")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inertia (suma de distancias al centroide)")
    ax1.set_title("Método del Codo")
    ax1.grid(True, alpha=0.3)

    ax2.plot(metrics.index, metrics["silhouette"], "o-", color="darkorange")
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silueta promedio (más alto = mejor)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_k_distance(X: np.ndarray, k: int = 5):
    """
    Curva k-distance para elegir ``eps`` en DBSCAN.

    La heurística clásica (Ester et al., 1996): graficar la distancia
    al k-ésimo vecino para cada punto, ordenada de menor a mayor.
    El "codo" de la curva sugiere un buen valor de ``eps``.
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # Distancia al k-ésimo vecino (la última columna)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_distances, color="seagreen")
    ax.set_xlabel("Puntos ordenados por distancia")
    ax.set_ylabel(f"Distancia al {k}º vecino más cercano")
    ax.set_title(
        f"Curva k-distance (k={k}) — el codo sugiere un valor adecuado de eps"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Clusters en espacio 2D (PCA)",
):
    """
    Scatter de los datos proyectados a 2D, coloreados por cluster.

    Los puntos con etiqueta ``-1`` (ruido en DBSCAN) se pintan en gris.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    unique = np.unique(labels)

    for c in unique:
        mask = labels == c
        if c == -1:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c="lightgray",
                s=15,
                alpha=0.5,
                label="Ruido",
            )
        else:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=25,
                alpha=0.7,
                label=f"Cluster {c}",
            )

    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_segment_radar(profiles: pd.DataFrame, features: Iterable[str]):
    """
    Radar chart con plotly para comparar perfiles de segmentos.

    Las features se normalizan a [0, 1] *entre clusters* para que el radar
    sea comparativo. Devuelve una figura plotly.

    Parameters
    ----------
    profiles : pd.DataFrame
        Tabla de perfiles (output de ``build_segment_profiles``).
    features : Iterable[str]
        Columnas a incluir en el radar (típicamente RFM + extras).
    """
    import plotly.graph_objects as go  # import perezoso para no forzar dependencia

    cols = [c for c in features if c in profiles.columns]
    valid = profiles.index != -1
    sub = profiles.loc[valid, cols].copy()

    # Min-max scaling por columna
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)

    fig = go.Figure()
    for cluster_id, row in sub_norm.iterrows():
        name = profiles.loc[cluster_id].get("segment_name", f"Cluster {cluster_id}")
        fig.add_trace(
            go.Scatterpolar(
                r=list(row.values) + [row.values[0]],
                theta=cols + [cols[0]],
                fill="toself",
                name=f"{cluster_id} · {name}",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Perfil comparativo de segmentos (normalizado)",
        showlegend=True,
    )
    return fig

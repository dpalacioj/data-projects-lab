"""
Algoritmos de clustering y métricas de evaluación.

Funciones de alto nivel para entrenar K-Means y DBSCAN, calcular métricas
internas de calidad, y buscar el K óptimo con el método del codo y silueta.

Notas didácticas
----------------
- **Inertia** (suma de distancias al centroide más cercano): siempre disminuye
  al aumentar K. Por eso se usa con el "método del codo" en lugar de
  buscar su mínimo absoluto.
- **Silhouette Score**: rango [-1, 1]. Cercano a 1 = clusters bien definidos,
  cercano a 0 = solapamiento, negativo = puntos mal asignados.
- **Davies-Bouldin Index**: cuanto MÁS BAJO mejor (mide la similitud entre
  cada cluster y el más parecido).
- **Calinski-Harabasz Index**: cuanto MÁS ALTO mejor (varianza entre clusters
  vs. varianza intra-cluster).

Ninguna métrica es perfecta. Combínalas con interpretación de negocio.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from segmentacion.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    **kwargs,
) -> KMeans:
    """Entrena un modelo K-Means y devuelve el estimador ajustado."""
    params = {"n_init": 10, "max_iter": 300, "random_state": random_state, **kwargs}
    model = KMeans(n_clusters=n_clusters, **params)
    model.fit(X)
    logger.info(
        f"K-Means k={n_clusters} entrenado | inertia={model.inertia_:.2f} | "
        f"iter={model.n_iter_}"
    )
    return model


def find_optimal_k(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula métricas de calidad para un rango de valores de K.

    Devuelve un DataFrame con columnas:
    ``k, inertia, silhouette, davies_bouldin, calinski_harabasz``.
    Útil para visualizar simultáneamente el codo, la silueta y demás.

    Parameters
    ----------
    X : np.ndarray
        Datos preprocesados (escalados).
    k_range : Iterable[int]
        Valores de k a probar.
    random_state : int
        Semilla.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for k in k_range:
        model = fit_kmeans(X, n_clusters=k, random_state=random_state)
        labels = model.labels_
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
            }
        )
    return pd.DataFrame(rows).set_index("k")


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def fit_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    **kwargs,
) -> DBSCAN:
    """
    Entrena DBSCAN y devuelve el estimador ajustado.

    Recordatorios pedagógicos:
    - ``eps``: radio del vecindario. Si es muy pequeño, casi todo se vuelve ruido.
      Si es muy grande, todo termina en un solo cluster.
    - ``min_samples``: mínimo de puntos para formar un núcleo. Una regla
      heurística común es ``min_samples >= 2 * n_features``.
    - Etiqueta ``-1``: punto considerado ruido (outlier).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, **kwargs)
    model.fit(X)

    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    logger.info(
        f"DBSCAN eps={eps} min_samples={min_samples} | "
        f"clusters={n_clusters} | ruido={n_noise} ({100 * n_noise / len(labels):.1f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Calcula métricas internas de clustering ignorando puntos de ruido.

    Para DBSCAN, los puntos con etiqueta ``-1`` se excluyen del cálculo,
    porque las métricas basadas en distancia no están bien definidas
    cuando hay una "clase basura".

    Parameters
    ----------
    X : np.ndarray
        Datos preprocesados.
    labels : np.ndarray
        Etiquetas asignadas por el modelo.

    Returns
    -------
    dict[str, float]
        Métricas: ``silhouette``, ``davies_bouldin``, ``calinski_harabasz``,
        ``n_clusters``, ``n_noise``.
    """
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int(np.sum(~mask))

    if n_clusters < 2:
        # Las métricas necesitan al menos 2 clusters reales.
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }

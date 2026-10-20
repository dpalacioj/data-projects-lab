"""
Demo interactiva de Segmentación de Clientes.

Permite a un usuario no técnico:
- Conocer el caso de negocio.
- Explorar las features RFM con visualizaciones.
- Probar K-Means o DBSCAN con sus propios hiperparámetros.
- Ver el perfil de cada segmento y la acción de marketing recomendada.

Ejecución::

    cd projects/segmentacion-clientes
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Permitir importar el paquete segmentacion aunque streamlit ejecute desde otra raiz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segmentacion.config import (  # noqa: E402
    EXTENDED_NUMERIC_FEATURES,
    FEATURES_DATA_FILE,
)
from segmentacion.features.preprocessing import build_preprocessing_pipeline  # noqa: E402
from segmentacion.models.clustering import evaluate_clustering, fit_dbscan, fit_kmeans  # noqa: E402
from segmentacion.models.profiling import build_segment_profiles, label_segments  # noqa: E402

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentación de Clientes",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data
def load_features() -> pd.DataFrame:
    """Carga el dataset de features por cliente."""
    if not FEATURES_DATA_FILE.exists():
        st.error(
            f"No se encontró {FEATURES_DATA_FILE.name}. "
            "Ejecuta los notebooks 01-03 antes de abrir la app."
        )
        st.stop()
    return pd.read_parquet(FEATURES_DATA_FILE)


@st.cache_data
def preprocess(features: pd.DataFrame) -> np.ndarray:
    """Aplica el pipeline de preprocesamiento."""
    pipeline = build_preprocessing_pipeline(
        numeric_features=EXTENDED_NUMERIC_FEATURES,
        use_log=True,
    )
    return pipeline.fit_transform(features[EXTENDED_NUMERIC_FEATURES])


# ---------------------------------------------------------------------------
# Sidebar — controles
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuración")
    section = st.radio(
        "Sección",
        options=[
            "📋 Caso de negocio",
            "🔍 Exploración",
            "🤖 Clustering",
            "📊 Perfiles y acciones",
        ],
    )

    st.markdown("---")
    st.caption(
        "Proyecto educativo · Online Retail II (UCI ML Repo). "
        "K-Means y DBSCAN sobre features RFM extendidas."
    )


# ---------------------------------------------------------------------------
# Carga única para todas las secciones
# ---------------------------------------------------------------------------
features = load_features()
X = preprocess(features)


# ===========================================================================
# Sección 1 — Caso de negocio
# ===========================================================================
if section == "📋 Caso de negocio":
    st.title("🎯 Segmentación de Clientes")
    st.markdown(
        """
Una empresa de retail online quiere **personalizar sus campañas de marketing**.
Sin segmentación, todos los clientes reciben el mismo mensaje — y eso es
ineficiente: un cliente VIP no necesita el mismo descuento que uno
inactivo, y un cliente nuevo no debería recibir un email de "te extrañamos".

### Preguntas de negocio

1. ¿Qué **tipos** de clientes existen? ¿Cuántos hay de cada tipo?
2. ¿Qué clientes generan **más valor** (Champions)?
3. ¿Qué clientes están **en riesgo** de irse?
4. ¿Qué clientes responderían bien a una **promoción agresiva**?
5. ¿Hay clientes con **comportamiento atípico** (mayoristas, fraudes)?

### El dataset

**Online Retail II** (UCI ML Repository) — transacciones reales de un
retailer británico de regalos *all-occasion* entre 2009-12-01 y 2011-12-09.

A partir de las transacciones construimos seis features por cliente:
        """
    )
    st.markdown(
        """
- **Recency**: días desde su última compra.
- **Frequency**: número de facturas distintas.
- **Monetary**: gasto total acumulado (GBP).
- **AvgTicket**: gasto promedio por factura.
- **ProductDiversity**: nº de productos únicos comprados.
- **AvgQuantity**: cantidad promedio de items por factura.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes únicos", f"{len(features):,}")
    col2.metric("Mediana Monetary (GBP)", f"{features['Monetary'].median():.0f}")
    col3.metric("Mediana Recency (días)", f"{features['Recency'].median():.0f}")


# ===========================================================================
# Sección 2 — Exploración
# ===========================================================================
elif section == "🔍 Exploración":
    st.title("🔍 Exploración de Features")
    st.markdown(
        "Inspecciona la distribución de cada variable para entender la "
        "estructura del problema antes de clusterizar."
    )

    feat = st.selectbox("Selecciona una feature:", EXTENDED_NUMERIC_FEATURES)
    use_log = st.checkbox("Aplicar transformación log1p", value=True)

    series = features[feat]
    if use_log:
        series = np.log1p(series)
        feat_label = f"log1p({feat})"
    else:
        feat_label = feat

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            series,
            nbins=50,
            title=f"Histograma · {feat_label}",
            labels={"value": feat_label},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_box = px.box(series, title=f"Boxplot · {feat_label}", points="outliers")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Estadísticas")
    st.dataframe(features[EXTENDED_NUMERIC_FEATURES].describe().T.round(2))

    st.markdown("### Correlación entre features")
    corr = features[EXTENDED_NUMERIC_FEATURES].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================================
# Sección 3 — Clustering
# ===========================================================================
elif section == "🤖 Clustering":
    st.title("🤖 Probar Clustering")
    st.markdown(
        "Selecciona un algoritmo, ajusta sus hiperparámetros y observa "
        "cómo cambian los clusters resultantes."
    )

    algo = st.radio("Algoritmo:", ["K-Means", "DBSCAN"], horizontal=True)

    if algo == "K-Means":
        k = st.slider("Número de clusters (k)", min_value=2, max_value=10, value=4)
        with st.spinner(f"Entrenando K-Means con k={k}..."):
            model = fit_kmeans(X, n_clusters=k)
            labels = model.labels_
        n_clusters = k
        n_noise = 0
    else:
        col1, col2 = st.columns(2)
        eps = col1.slider("eps (radio del vecindario)", 0.1, 2.0, 0.5, 0.05)
        min_samples = col2.slider("min_samples", 2, 30, 2 * X.shape[1])
        with st.spinner(f"Entrenando DBSCAN eps={eps} min_samples={min_samples}..."):
            model = fit_dbscan(X, eps=eps, min_samples=min_samples)
            labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

    # Métricas
    metrics = evaluate_clustering(X, labels)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters", n_clusters)
    col2.metric("Outliers", n_noise)
    col3.metric(
        "Silhouette",
        f"{metrics['silhouette']:.3f}" if not np.isnan(metrics["silhouette"]) else "N/A",
    )
    col4.metric(
        "Davies-Bouldin",
        f"{metrics['davies_bouldin']:.3f}" if not np.isnan(metrics["davies_bouldin"]) else "N/A",
    )

    # Visualización 2D con PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    plot_df = pd.DataFrame({"PC1": X_2d[:, 0], "PC2": X_2d[:, 1], "cluster": labels.astype(str)})

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title=f"Clientes en espacio 2D (PCA · varianza explicada {pca.explained_variance_ratio_.sum():.1%})",
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribución por cluster
    st.markdown("### Distribución de clientes por cluster")
    counts = pd.Series(labels).value_counts().sort_index()
    fig_bar = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={"x": "Cluster", "y": "Nº clientes"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Guardar labels en session_state para reutilizar en la sección de perfiles
    st.session_state["last_labels"] = labels
    st.session_state["last_algo"] = algo


# ===========================================================================
# Sección 4 — Perfiles y acciones
# ===========================================================================
elif section == "📊 Perfiles y acciones":
    st.title("📊 Perfiles de Segmentos y Acciones de Marketing")

    if "last_labels" not in st.session_state:
        st.warning(
            "Primero ejecuta un modelo en la sección **🤖 Clustering** para ver el perfilamiento."
        )
        st.stop()

    labels = st.session_state["last_labels"]
    algo = st.session_state["last_algo"]
    st.caption(f"Mostrando perfiles del último modelo: **{algo}**.")

    profiles = build_segment_profiles(features[EXTENDED_NUMERIC_FEATURES], labels=labels)
    profiles = label_segments(profiles)

    # Tabla resumen
    st.markdown("### Tabla resumen")
    cols_to_show = ["n_customers", "share", "segment_name"] + EXTENDED_NUMERIC_FEATURES + ["action"]
    st.dataframe(profiles[cols_to_show], use_container_width=True)

    # Gráficos comparativos por feature
    st.markdown("### Promedio por cluster")
    feat = st.selectbox("Variable:", EXTENDED_NUMERIC_FEATURES, key="prof_feat")
    fig = px.bar(
        profiles.reset_index(),
        x="cluster",
        y=feat,
        color="segment_name",
        text_auto=".2f",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recomendaciones por cluster
    st.markdown("### Acciones de marketing recomendadas")
    for cluster_id, row in profiles.iterrows():
        with st.expander(
            f"Cluster {cluster_id} · {row['segment_name']} · {int(row['n_customers'])} clientes"
        ):
            st.markdown(f"**Acción recomendada:** {row['action']}")
            st.markdown(
                f"- Recency promedio: **{row['Recency']:.0f}** días\n"
                f"- Frequency promedio: **{row['Frequency']:.1f}** facturas\n"
                f"- Monetary promedio: **{row['Monetary']:,.0f}** GBP\n"
                f"- Ticket promedio: **{row['AvgTicket']:,.0f}** GBP"
            )

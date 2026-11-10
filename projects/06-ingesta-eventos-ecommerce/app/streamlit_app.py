"""Dashboard Streamlit: lee de la base relacional y muestra KPIs.

Correr desde la carpeta del proyecto:
    uv run streamlit run app/streamlit_app.py

O con make:
    make serve
"""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingesta_eventos.db import crear_engine, probar_conexion  # noqa: E402
from ingesta_eventos.queries import (  # noqa: E402
    clientes_recurrentes,
    conteos_tablas,
    ingresos_por_canal,
    ingresos_por_dia,
    ingresos_por_pais,
    top_productos,
)

st.set_page_config(page_title="Dashboard eventos e-commerce", page_icon=":bar_chart:", layout="wide")

st.title("Dashboard de eventos de e-commerce")
st.caption(
    "Datos leidos de la base relacional cargada por el pipeline ETL. "
    "Los JSON crudos viven en `data/raw/`; aqui ya estan normalizados."
)

# ---------------------------------------------------------------------------
# Conexion (cacheada; Streamlit reusa el engine entre reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def _engine():
    return crear_engine()


try:
    engine = _engine()
    info = probar_conexion(engine)
except Exception as exc:  # noqa: BLE001
    st.error(f"No se pudo conectar a la base: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Conexion")
    st.code(info["url"], language="text")
    st.caption(f"driver: `{info['driver']}`")
    st.caption("La URL se arma en `config.py` desde `.env`. La password (si la hay) queda oculta.")

# ---------------------------------------------------------------------------
# Conteos: verifica que hay datos
# ---------------------------------------------------------------------------
conteos = conteos_tablas(engine)

if conteos["pedidos"] == 0:
    st.warning(
        "La base esta vacia. Ejecuta primero el pipeline:\n\n"
        "```bash\n"
        "make generate-data   # genera los JSON sinteticos\n"
        "make load            # los carga en la base\n"
        "```"
    )
    st.stop()

st.subheader("Conteos por tabla")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{conteos['clientes']:,}")
c2.metric("Productos", f"{conteos['productos']:,}")
c3.metric("Pedidos", f"{conteos['pedidos']:,}")
c4.metric("Items", f"{conteos['items_pedido']:,}")

st.divider()

# ---------------------------------------------------------------------------
# Serie temporal
# ---------------------------------------------------------------------------
st.subheader("Ingresos y pedidos por dia")
df_tiempo = ingresos_por_dia(engine)
col_a, col_b = st.columns(2)
with col_a:
    fig = px.line(df_tiempo, x="dia", y="ingresos", markers=True, title="Ingresos diarios")
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    fig = px.bar(df_tiempo, x="dia", y="pedidos", title="Pedidos diarios")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Canal y pais
# ---------------------------------------------------------------------------
st.subheader("Reparto por canal y por pais")
col_a, col_b = st.columns(2)
with col_a:
    df_canal = ingresos_por_canal(engine)
    fig = px.pie(df_canal, names="canal", values="ingresos", title="Ingresos por canal")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_canal, use_container_width=True, hide_index=True)
with col_b:
    df_pais = ingresos_por_pais(engine)
    fig = px.bar(df_pais, x="pais", y="ingresos", title="Ingresos por pais", text_auto=".2s")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_pais, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Top productos y clientes
# ---------------------------------------------------------------------------
st.subheader("Top productos y clientes recurrentes")
limite = st.slider("Limite", 5, 50, 10)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Top productos por ingresos**")
    df_prod = top_productos(engine, limite=limite)
    st.dataframe(df_prod, use_container_width=True, hide_index=True)
with col_b:
    st.markdown("**Clientes con mas pedidos**")
    df_cli = clientes_recurrentes(engine, limite=limite)
    st.dataframe(df_cli, use_container_width=True, hide_index=True)

st.caption(
    "Las queries estan en `src/ingesta_eventos/queries.py`. "
    "Si necesitas una nueva, anadela ahi y llamala desde esta app."
)

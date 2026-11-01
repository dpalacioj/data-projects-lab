"""Interfaz Streamlit para el usuario de negocio (Unidad 4).

Toda la logica esta en `segmentacion.servicio.Predictor`; aqui solo hay
pantalla. Correr desde la carpeta del proyecto:

    uv run streamlit run app/recompra_streamlit.py     -> http://localhost:8501
"""

import sys
from pathlib import Path

import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentacion.servicio import Predictor  # noqa: E402

st.set_page_config(page_title="Recompra", page_icon="🛒")  # primera instruccion de Streamlit


@st.cache_resource  # el modelo se carga una vez, no en cada interaccion
def obtener_predictor() -> Predictor:
    return Predictor.desde_carpeta()


predictor = obtener_predictor()
meta = predictor.metadatos

st.title("¿Quién volverá a comprar en los próximos 90 días?")

with st.sidebar:
    st.caption(
        f"Modelo {predictor.version_modelo} · corte {meta['fecha_corte']} · "
        f"AUC validación {meta['auc_validacion']}"
    )
    umbrales = [fila["umbral"] for fila in meta["tabla_umbrales"]]
    umbral = st.select_slider("Umbral de decisión", options=umbrales, value=predictor.umbral_acordado)
    fila = next(f for f in meta["tabla_umbrales"] if f["umbral"] == umbral)
    st.metric("Clientes señalados (cohorte de prueba)", fila["n_senialados"])
    st.metric("Precisión", fila["precision"])
    st.metric("Recall", fila["recall"])
    st.caption(
        "Bajar el umbral contacta a más clientes y encuentra más recompras, "
        "con más contactos innecesarios."
    )

uno, lista = st.tabs(["Un cliente", "Lista de clientes"])

with uno:
    with st.form("cliente"):
        c1, c2 = st.columns(2)
        entrada = {
            "monetario": c1.number_input("Gasto total (£)", min_value=0.0, value=900.0),
            "frecuencia": c1.number_input("Facturas", min_value=1, value=3),
            "productos_distintos": c1.number_input("Productos distintos", min_value=1, value=25),
            "recencia": c2.number_input("Días desde la última compra", min_value=0, value=60),
            "antiguedad": c2.number_input("Días desde la primera compra", min_value=0, value=300),
            "tasa_devolucion": c2.number_input("Tasa de devolución", min_value=0.0, value=0.0),
            "pais": st.selectbox("País", ["United Kingdom", "Otro país"]),
        }
        enviado = st.form_submit_button("Predecir")
    if enviado:
        r = predictor.predecir(entrada, umbral=umbral)
        st.metric("Probabilidad de recompra", f"{r['probabilidad']:.0%}")
        (st.success if r["recompra"] else st.info)("Contactar" if r["recompra"] else "No contactar")

with lista:
    archivo = st.file_uploader("CSV con una fila por cliente y las columnas del modelo", type="csv")
    if archivo:
        try:
            df = pl.read_csv(archivo)
            resultado = predictor.predecir_lote(df, umbral=umbral)
        except Exception as error:  # mensaje claro, no un traceback
            st.error(f"No pude procesar el archivo: {error}")
        else:
            st.dataframe(resultado)
            contactar = resultado.filter(pl.col("recompra"))
            st.download_button(
                f"Descargar {contactar.height} clientes a contactar",
                contactar.write_csv(),
                "clientes_a_contactar.csv",
                "text/csv",
            )

st.caption("Datos: Online Retail II (Chen, 2019), CC BY 4.0.")

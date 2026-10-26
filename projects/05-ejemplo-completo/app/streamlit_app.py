"""Interfaz Streamlit: formulario con las features, muestra prediccion.

Correr desde la carpeta del proyecto:
    uv run streamlit run app/streamlit_app.py

O con make:
    make serve-streamlit
"""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ejemplo_completo.config import CLASES, FEATURES_NUM, MODEL_PATH  # noqa: E402
from ejemplo_completo.data import cargar  # noqa: E402
from ejemplo_completo.predict import predecir  # noqa: E402

st.set_page_config(page_title="Clasificador de vinos", page_icon="🍷")
st.title("Clasificador de cultivares de vino")
st.caption(
    "Dataset `load_wine` de scikit-learn. 3 cultivares, 13 features numericas."
)

if not MODEL_PATH.exists():
    st.error(
        f"No se encontro el modelo en `{MODEL_PATH}`.\n\n"
        "Entrena primero desde la carpeta del proyecto:\n\n"
        "```bash\n"
        "make train\n"
        "```\n\n"
        "o directamente:\n\n"
        "```bash\n"
        "uv run python -m ejemplo_completo.train\n"
        "```"
    )
    st.stop()


df = cargar()
muestra_media = df[FEATURES_NUM].mean()

st.subheader("Caracteristicas de la muestra")
st.caption("Rangos y valores por defecto tomados del promedio del dataset.")

cols = st.columns(2)
valores: dict[str, float] = {}
for i, feature in enumerate(FEATURES_NUM):
    col = cols[i % 2]
    min_v = float(df[feature].min())
    max_v = float(df[feature].max())
    default = float(muestra_media[feature])
    valores[feature] = col.number_input(
        feature,
        min_value=min_v,
        max_value=max_v,
        value=default,
        step=(max_v - min_v) / 100 if max_v > min_v else 0.01,
        format="%.4f",
    )

st.divider()

if st.button("Predecir cultivar", type="primary"):
    resultado = predecir(valores)
    clase = resultado["clase"]
    probs = resultado["probabilidades"]

    st.success(f"Cultivar predicho: **{clase}**")

    st.subheader("Probabilidades por clase")
    for c in CLASES:
        p = probs[c]
        st.write(f"**{c}** — {p:.1%}")
        st.progress(p)

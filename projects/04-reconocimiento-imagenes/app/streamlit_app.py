"""Interfaz Streamlit: sube una imagen, muestra las top-3 clases predichas.

Correr:
    uv run streamlit run projects/04-reconocimiento-imagenes/app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconocimiento_imagenes.config import CLASES, MODEL_PATH  # noqa: E402
from reconocimiento_imagenes.predict import cargar_modelo, predecir  # noqa: E402

st.set_page_config(page_title="Reconocimiento de prendas", page_icon="👕")
st.title("Reconocimiento de prendas (Fashion-MNIST)")
st.caption(f"10 clases: {', '.join(CLASES)}")


@st.cache_resource
def obtener_modelo():
    if not MODEL_PATH.exists():
        return None
    return cargar_modelo()


modelo = obtener_modelo()
if modelo is None:
    st.error(
        f"No se encontro el modelo en `{MODEL_PATH}`.\n\n"
        "Entrena primero con:\n\n"
        "```bash\n"
        "uv run python -m reconocimiento_imagenes.train\n"
        "```"
    )
    st.stop()

archivo = st.file_uploader(
    "Sube una imagen (jpg, png, bmp). Se convertira a grises y 28x28.",
    type=["jpg", "jpeg", "png", "bmp"],
)

if archivo is not None:
    img = Image.open(archivo)
    col_img, col_preds = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Entrada", use_container_width=True)
    with col_preds:
        st.subheader("Top-3 predicciones")
        for clase, prob in predecir(modelo, img, topk=3):
            st.write(f"**{clase}** · {prob:.1%}")
            st.progress(prob)
else:
    st.info("Esperando una imagen...")

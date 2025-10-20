from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load
# import pickle
from sklearn.pipeline import Pipeline

# ============================================================
# Cómo ejecutar la app:
# Desde la terminal, ubicándote en la raíz del proyecto, usa:

# Opción 1: Usa uv run (recomendado)
# uv run streamlit run projects/titanic/08_titanic_streamlit.py

# Opción 2: Activa el entorno virtual primero
# source .venv/bin/activate
# streamlit run projects/titanic/08_titanic_streamlit.py
# ============================================================


def get_user_data() -> pd.DataFrame:
    """
    Recoge los datos ingresados por el usuario a través de la interfaz de Streamlit,
    los preprocesa y retorna un DataFrame listo para alimentar el modelo.
    """
    user_data = {}

    # Dividir la pantalla en dos columnas para ingresar datos numéricos
    col_a, col_b = st.columns(2)
    with col_a:
        user_data["age"] = st.number_input(
            label="Edad:", min_value=0, max_value=100, value=20, step=1
        )
        user_data["sibsp"] = st.slider(
            label="Número de hermanos y cónyuges a bordo:",
            min_value=0, max_value=15, value=3, step=1,
        )
    with col_b:
        user_data["fare"] = st.number_input(
            label="Costo del boleto:",
            min_value=0, max_value=300, value=80, step=1,
        )
        user_data["parch"] = st.slider(
            label="Número de padres e hijos a bordo:",
            min_value=0, max_value=15, value=3, step=1,
        )

    # Dividir en tres columnas para seleccionar opciones categóricas
    col1, col2, col3 = st.columns(3)
    with col1:
        user_data["pclass"] = st.radio(
            label="Clase del boleto:", options=["1st", "2nd", "3rd"], horizontal=False
        )
    with col2:
        user_data["sex"] = st.radio(
            label="Sexo:", options=["Woman", "Man"], horizontal=False
        )
    with col3:
        user_data["embarked"] = st.radio(
            label="Puerto de embarque:",
            options=["Cherbourg", "Queenstown", "Southampton"],
            index=1,
        )


    # Convertir el diccionario a DataFrame y transponerlo para tener una fila con todas las variables
    df = pd.DataFrame.from_dict(user_data, orient="index").T

    # Preprocesamiento: mapear los valores de texto a los formatos esperados por el modelo
    df["sex"] = df["sex"].map({"Man": "male", "Woman": "female"})
    df["pclass"] = df["pclass"].map({"1st": 1, "2nd": 2, "3rd": 3})
    df["embarked"] = df["embarked"].map({
        "Cherbourg": "C",
        "Queenstown": "Q",
        "Southampton": "S",
    })

    return df


## Este método se usa si se quiere cargar el modelo en `.joblib`

@st.cache_resource
def load_model(model_file_path: Path) -> Pipeline:
    """
    Carga un modelo guardado en formato joblib (.joblib).
    Se usa un spinner para indicar que se está cargando el modelo.
    """
    with st.spinner("Cargando modelo..."):
        model = load(model_file_path)
    return model

# @st.cache_resource
# def load_model_pickle(model_file_path: str) -> Pipeline:
#     """
#     Carga un modelo guardado en formato pickle (.pickle).
#     Se usa un spinner para indicar que se está cargando el modelo.
#     """
#     with st.spinner("Cargando modelo..."):
#         with open(model_file_path, "rb") as f:
#             model = pickle.load(f)
#     return model


def main() -> None:
    # Nombre del modelo que vamos a usar
    model_name = "titanic_classification-random_forest-v1.joblib"

    # Obtener el directorio actual donde está este archivo .py
    CURRENT_DIR = Path(__file__).parent

    # Construir la ruta al modelo (está en la carpeta "modelos" dentro de 'titanic')
    MODELS_DIR = CURRENT_DIR / "modelos"
    model_path = MODELS_DIR / model_name

    # Verificar que el modelo existe
    if not model_path.exists():
        st.error(f"❌ No se encontró el modelo en: {model_path}")
        st.stop()

    # Mostrar la imagen del Titanic si existe
    IMAGES_DIR = CURRENT_DIR / "imagenes"
    image_path = IMAGES_DIR / "titanic.jpg"
    if image_path.exists():
        st.image(str(image_path), caption="Esto fue el Titanic")

    # Título de la aplicación
    st.header("¿Sobrevivirías al Titanic? 🚢")
    
    # Recoger los datos del usuario
    df_user_data = get_user_data()

    # Cargar el modelo
    model = load_model(model_path)
    #model = load_model_pickle(model_file_path=model_path)
    
    # Realizar la predicción con los datos del usuario
    state = model.predict(df_user_data)[0]
    
    # Definir emojis para la visualización
    emojis = ["😕", "😀"]
    
    st.write("")
    st.title(f"Chance to survive: {emojis[state]}")
    
    if state == 0:
        st.error("¡Mala noticia, amigo! ¡Serás comida de tiburones! 🦈")
    else:
        st.success("¡Felicidades! ¡Puedes estar tranquilo, sobrevivirías al Titanic y ganarás el curso de Analítica de Datos! 🤩")
    
if __name__ == "__main__":
    main()

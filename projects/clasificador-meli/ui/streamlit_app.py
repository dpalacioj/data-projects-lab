import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
import json
import ast
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para imports relativos
# ui/streamlit_app.py -> projects/clasificador-meli/ui/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.preprocessing import Preprocessing
from src.config import MODEL_PATH, MODEL_NAME

st.set_page_config(page_title="Clasificación de Productos", layout="wide")
st.title(":package: Clasificador de Condición del Producto")
st.markdown("Sube tus datos o ingrésalos manualmente para obtener una predicción.")

@st.cache_resource
def load_model(path_model):
    return load(path_model)

# Cargar modelo
model_loaded = os.path.join(MODEL_PATH, MODEL_NAME)
model = load_model(model_loaded)
preprocessor = Preprocessing()



# Input: archivo o manual
option = st.radio(":question: ¿Cómo quieres ingresar los datos?", ["Subir CSV", "Subir JSON" ,"Ingreso Manual"])

if option == "Subir CSV":
    uploaded_file = st.file_uploader(":green_book: Sube tu archivo CSV con estructura original", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"Archivo cargado con shape: {df_raw.shape}")
            st.dataframe(df_raw.head())

            # Preprocesar
            df_raw_copy = df_raw.copy()
            df_processed = preprocessor.transform(df_raw_copy)
            df_processed = df_processed.drop(columns=["condition"])

            # Predicción
            preds = model.predict(df_processed)
            df_result = df_raw.copy()
            df_result["predicted_condition"] = preds
            df_result["predicted_condition"] = df_result["predicted_condition"].map({0: "new", 1: "used"})

            st.subheader("Predicciones")
            st.dataframe(df_result)

            # Descargar resultado
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar resultados", csv, "predicciones.csv", "text/csv")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

# --- Opción 2: Cargar JSON ---
elif option == "Subir JSON":
    uploaded_json = st.file_uploader(":page_with_curl: Sube un archivo JSON con estructura de entrada original", type=["jsonlines"])
    if uploaded_json:
        try:
            data = [json.loads(line) for line in uploaded_json]

            df_raw = pd.DataFrame(data)

            st.success(f"Archivo JSON cargado: {df_raw.shape}")
            st.dataframe(df_raw.head())

            # Preprocesar
            df_raw_copy = df_raw.copy()
            df_processed = preprocessor.transform(df_raw_copy)
            df_processed = df_processed.drop(columns=["condition"])

            # Predicción
            CONDITION_MAP = {0: "new", 1: "used"}
            preds = model.predict(df_processed)

            df_result = df_raw.copy()
            translated_preds = [CONDITION_MAP.get(p, "Desconocido") for p in preds]

            df_result = df_raw.copy()
            df_result["predicted_condition"] = translated_preds

            st.subheader("Predicciones")
            st.dataframe(df_result)

            json_output = df_result.to_json(orient="records", indent=2)
            st.download_button("Descargar resultados", json_output, "predicciones.json", "application/json")

        except Exception as e:
            st.error(f"Error procesando JSON: {e}")

elif option == "Ingreso Manual":
    st.markdown(":fist: Pega el JSON completo de una publicación (como en el dataset original):")

    user_input = st.text_area("JSON de producto", height=300, placeholder="Pega aquí el JSON...")

    if st.button("Predecir"):
        try:
            # Parsear el JSON del usuario
            input_json = json.loads(user_input)

            # Soportar tanto objeto individual como lista con 'inputs'
            if isinstance(input_json, dict) and "inputs" in input_json:
                df_raw = pd.DataFrame(input_json["inputs"])
            elif isinstance(input_json, list):
                df_raw = pd.DataFrame(input_json)
            else:
                df_raw = pd.DataFrame([input_json])

            st.success(f"JSON válido. Filas cargadas: {df_raw.shape[0]}")
            st.dataframe(df_raw.head())

            # Preprocesar
            df_raw_copy = df_raw.copy()
            df_processed = preprocessor.transform(df_raw_copy)
            df_processed = df_processed.drop(columns=["condition"])

            # Predicción
            CONDITION_MAP = {0: "new", 1: "used"}
            preds = model.predict(df_processed)
            translated_preds = [CONDITION_MAP.get(p, "Desconocido") for p in preds]

            df_result = df_raw.copy()
            df_result["predicted_condition"] = translated_preds

            st.subheader("Resultado")
            st.dataframe(df_result)

            json_result = df_result.to_json(orient="records", indent=2)
            st.download_button("Descargar predicción JSON", json_result, "prediccion.json", "application/json")

        except Exception as e:
            st.error(f"Error al procesar el JSON: {e}")

"""
UI de Streamlit para el Clasificador de Productos MercadoLibre.

Permite cargar datos en múltiples formatos y obtener predicciones
sobre si un producto es 'new' o 'used'.
"""
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json
import sys
from pathlib import Path

# Configurar path para imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.preprocessing import Preprocessing
from src.config import MODEL_PATH, MODEL_NAME
from src.utils.logger import setup_logger

# Configurar logger para la UI
logger = setup_logger(__name__, level=20)  # INFO level para UI

# Configuración de página
st.set_page_config(
    page_title="Clasificador MercadoLibre",
    page_icon=":package:",
    layout="wide"
)

# Header
st.title(":package: Clasificador de Condición del Producto")
st.markdown("""
Sube tus datos en diferentes formatos para obtener predicciones sobre si los productos son **nuevos** o **usados**.
""")

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    ### Formatos soportados:
    - **CSV**: Archivo con estructura completa
    - **JSON**: Array de objetos JSON
    - **JSONLINES**: Un JSON por línea
    - **Parquet**: Datos preprocesados
    - **Manual**: Copiar/pegar JSON

    ### Archivos de prueba:
    Usa los archivos en `test_data/` para probar la UI.
    """)


@st.cache_resource
def load_model_cached(path_model):
    """Carga el modelo desde disco (con cache)."""
    logger.info(f"Cargando modelo desde: {path_model}")
    model = load(path_model)
    logger.info("Modelo cargado exitosamente")
    return model


def add_predictions_with_probs(df_raw, df_processed, model):
    """
    Agrega predicciones y probabilidades al DataFrame.

    Args:
        df_raw: DataFrame original
        df_processed: DataFrame preprocesado (sin 'condition')
        model: Modelo entrenado

    Returns:
        DataFrame con columnas adicionales de predicción
    """
    logger.info(f"Generando predicciones para {len(df_processed)} productos")

    # Predecir
    preds = model.predict(df_processed)
    probs = model.predict_proba(df_processed)

    # Crear DataFrame de resultados
    df_result = df_raw.copy()
    df_result["predicted_condition"] = preds
    df_result["predicted_condition"] = df_result["predicted_condition"].map({0: "new", 1: "used"})

    # Agregar probabilidades
    df_result["probability_new"] = probs[:, 0]
    df_result["probability_used"] = probs[:, 1]
    df_result["confidence"] = np.max(probs, axis=1)  # Máxima probabilidad

    # Log de distribución de predicciones
    count_new = (df_result["predicted_condition"] == "new").sum()
    count_used = (df_result["predicted_condition"] == "used").sum()
    logger.info(f"Predicciones: {count_new} NEW | {count_used} USED | Confianza promedio: {df_result['confidence'].mean():.2%}")

    return df_result


def display_results(df_result, show_distribution_chart=True):
    """
    Muestra los resultados de predicción de forma estandarizada.

    Esta función unifica la visualización para todas las opciones de entrada,
    mostrando siempre el mismo formato: métricas, tabla con gradient, y gráfico opcional.

    Args:
        df_result: DataFrame con predicciones y probabilidades
        show_distribution_chart: Si True, muestra gráfico de distribución de predicciones
    """
    logger.info(f"Mostrando resultados para {len(df_result)} productos")

    # Mensaje de éxito
    st.success(f"✅ Predicciones generadas para {len(df_result)} productos")

    # Métricas resumidas (3 columnas)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Total Productos", len(df_result))
    with col2:
        count_new = (df_result["predicted_condition"] == "new").sum()
        percentage_new = (count_new / len(df_result)) * 100
        st.metric("🆕 Predichos como NEW", f"{count_new} ({percentage_new:.1f}%)")
    with col3:
        count_used = (df_result["predicted_condition"] == "used").sum()
        percentage_used = (count_used / len(df_result)) * 100
        st.metric("♻️ Predichos como USED", f"{count_used} ({percentage_used:.1f}%)")

    # Gráfico de distribución (opcional)
    if show_distribution_chart:
        st.subheader("📊 Distribución de Predicciones")
        chart_data = pd.DataFrame({
            'Condición': ['NEW', 'USED'],
            'Cantidad': [count_new, count_used]
        })
        st.bar_chart(chart_data.set_index('Condición'))

    # Tabla de resultados con gradient
    st.subheader("📋 Resultados Detallados")

    # Seleccionar columnas a mostrar
    display_cols = ["predicted_condition", "confidence", "probability_new", "probability_used"]

    # Agregar columnas adicionales si existen (price, title, etc.)
    optional_cols = ["title", "price", "seller_id", "listing_type_id"]
    for col in optional_cols:
        if col in df_result.columns:
            display_cols.insert(0, col)

    # Mostrar solo columnas que existen
    existing_display_cols = [col for col in display_cols if col in df_result.columns]

    st.dataframe(
        df_result[existing_display_cols].style.background_gradient(
            subset=["confidence"],
            cmap="Greens"
        ),
        use_container_width=True,
        height=400
    )

    # Estadísticas adicionales
    with st.expander("📈 Ver estadísticas adicionales"):
        st.write("**Confianza promedio por condición:**")
        stats = df_result.groupby("predicted_condition")["confidence"].agg(['mean', 'min', 'max'])
        stats.columns = ['Promedio', 'Mínima', 'Máxima']
        st.dataframe(stats.style.format("{:.2%}"))

    return df_result


# Cargar modelo
try:
    model_file = MODEL_PATH / MODEL_NAME
    model = load_model_cached(model_file)
    preprocessor = Preprocessing()
    st.success("✅ Modelo cargado correctamente")
    logger.info("Aplicación iniciada correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}", exc_info=True)
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Selector de opciones
option = st.radio(
    "📂 ¿Cómo quieres ingresar los datos?",
    ["Subir CSV", "Subir JSON", "Subir JSONLINES", "Subir Parquet", "Ingreso Manual"],
    horizontal=True
)

# =============================================================================
# OPCIÓN 1: CSV
# =============================================================================
if option == "Subir CSV":
    st.subheader("📄 Subir archivo CSV")
    uploaded_file = st.file_uploader(
        "Sube un archivo CSV con la estructura completa del dataset",
        type=["csv"],
        help="El CSV debe contener todas las columnas del dataset original"
    )

    if uploaded_file is not None:
        try:
            logger.info(f"Usuario cargó archivo CSV: {uploaded_file.name}")
            with st.spinner("Procesando datos..."):
                # Cargar CSV
                df_raw = pd.read_csv(uploaded_file)
                logger.info(f"CSV cargado | Shape: {df_raw.shape}")
                st.info(f"📊 Datos cargados: **{df_raw.shape[0]} filas** x **{df_raw.shape[1]} columnas**")

                # Mostrar vista previa
                with st.expander("👀 Ver datos crudos"):
                    st.dataframe(df_raw.head(10))

                # Preprocesar
                df_processed = preprocessor.transform(df_raw.copy())
                if 'condition' in df_processed.columns:
                    df_processed = df_processed.drop(columns=["condition"])

                # Predecir
                df_result = add_predictions_with_probs(df_raw, df_processed, model)

                # Mostrar resultados (función estandarizada)
                display_results(df_result, show_distribution_chart=True)

                # Descargar
                csv_output = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar resultados CSV",
                    csv_output,
                    "predicciones.csv",
                    "text/csv",
                    use_container_width=True
                )

        except Exception as e:
            logger.error(f"Error al procesar datos: {e}", exc_info=True)
            st.error(f"❌ Error: {e}")
            with st.expander("Ver detalles del error"):
                st.exception(e)

# =============================================================================
# OPCIÓN 2: JSON
# =============================================================================
elif option == "Subir JSON":
    st.subheader("📄 Subir archivo JSON")
    uploaded_file = st.file_uploader(
        "Sube un archivo JSON (array de objetos)",
        type=["json"],
        help="Formato: [{...}, {...}, ...]"
    )

    if uploaded_file is not None:
        try:
            logger.info(f"Usuario cargó archivo JSON: {uploaded_file.name}")
            with st.spinner("Procesando datos..."):
                # Cargar JSON
                data = json.load(uploaded_file)

                # Convertir a DataFrame
                if isinstance(data, list):
                    df_raw = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df_raw = pd.DataFrame([data])
                else:
                    st.error("❌ Formato JSON no soportado")
                    st.stop()

                st.info(f"📊 Datos cargados: **{df_raw.shape[0]} productos**")

                # Preprocesar y predecir
                df_processed = preprocessor.transform(df_raw.copy())
                if 'condition' in df_processed.columns:
                    df_processed = df_processed.drop(columns=["condition"])

                df_result = add_predictions_with_probs(df_raw, df_processed, model)

                # Mostrar resultados (función estandarizada)
                display_results(df_result, show_distribution_chart=True)

                # Descargar
                json_output = df_result.to_json(orient="records", indent=2)
                st.download_button(
                    "⬇️ Descargar resultados JSON",
                    json_output,
                    "predicciones.json",
                    "application/json",
                    use_container_width=True
                )

        except Exception as e:
            logger.error(f"Error al procesar datos: {e}", exc_info=True)
            st.error(f"❌ Error: {e}")
            with st.expander("Ver detalles del error"):
                st.exception(e)

# =============================================================================
# OPCIÓN 3: JSONLINES
# =============================================================================
elif option == "Subir JSONLINES":
    st.subheader("📄 Subir archivo JSONLINES")
    uploaded_file = st.file_uploader(
        "Sube un archivo JSONLINES (un JSON por línea)",
        type=["jsonlines", "jsonl"],
        help="Formato: Un objeto JSON por línea"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Procesando datos..."):
                # Cargar JSONLINES
                data = [json.loads(line) for line in uploaded_file]
                df_raw = pd.DataFrame(data)

                st.info(f"📊 Datos cargados: **{df_raw.shape[0]} productos**")

                # Preprocesar y predecir
                df_processed = preprocessor.transform(df_raw.copy())
                if 'condition' in df_processed.columns:
                    df_processed = df_processed.drop(columns=["condition"])

                df_result = add_predictions_with_probs(df_raw, df_processed, model)

                # Mostrar resultados (función estandarizada)
                display_results(df_result, show_distribution_chart=True)

                # Descargar
                st.download_button(
                    "⬇️ Descargar resultados JSON",
                    df_result.to_json(orient="records", indent=2),
                    "predicciones.json",
                    "application/json",
                    use_container_width=True
                )

        except Exception as e:
            logger.error(f"Error al procesar datos: {e}", exc_info=True)
            st.error(f"❌ Error: {e}")
            with st.expander("Ver detalles del error"):
                st.exception(e)

# =============================================================================
# OPCIÓN 4: PARQUET
# =============================================================================
elif option == "Subir Parquet":
    st.subheader("📄 Subir archivo Parquet")
    st.warning("⚠️ **Nota:** El parquet debe estar SIN preprocesar (datos crudos) o preprocesado correctamente.")

    uploaded_file = st.file_uploader(
        "Sube un archivo Parquet",
        type=["parquet"],
        help="Archivo Parquet con datos del dataset"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Procesando datos..."):
                # Cargar Parquet
                df_raw = pd.read_parquet(uploaded_file)

                st.info(f"📊 Datos cargados: **{df_raw.shape[0]} filas** x **{df_raw.shape[1]} columnas**")

                # Verificar si ya está procesado (tiene 22-23 columnas)
                if df_raw.shape[1] <= 23 and 'seller_address' not in df_raw.columns:
                    st.info("🔄 Parquet parece estar preprocesado, usando directamente...")
                    df_processed = df_raw.copy()
                    if 'condition' in df_processed.columns:
                        df_processed = df_processed.drop(columns=["condition"])
                    df_result_base = df_raw.copy()
                else:
                    st.info("🔄 Preprocesando datos...")
                    df_processed = preprocessor.transform(df_raw.copy())
                    if 'condition' in df_processed.columns:
                        df_processed = df_processed.drop(columns=["condition"])
                    df_result_base = df_raw.copy()

                # Predecir
                df_result = add_predictions_with_probs(df_result_base, df_processed, model)

                # Mostrar resultados (función estandarizada)
                display_results(df_result, show_distribution_chart=True)

                # Descargar
                st.download_button(
                    "⬇️ Descargar resultados Parquet",
                    df_result.to_parquet(index=False),
                    "predicciones.parquet",
                    use_container_width=True
                )

        except Exception as e:
            logger.error(f"Error al procesar datos: {e}", exc_info=True)
            st.error(f"❌ Error: {e}")
            with st.expander("Ver detalles del error"):
                st.exception(e)

# =============================================================================
# OPCIÓN 5: INGRESO MANUAL
# =============================================================================
elif option == "Ingreso Manual":
    st.subheader("✍️ Ingreso Manual de JSON")
    st.markdown("Pega un JSON completo de un producto (o array de productos):")

    user_input = st.text_area(
        "JSON del producto",
        height=300,
        placeholder='{"seller_address": {...}, "price": 100, ...}'
    )

    if st.button("🔮 Predecir", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Por favor ingresa un JSON válido")
        else:
            try:
                logger.info("Usuario ingresó JSON manualmente")
                with st.spinner("Procesando..."):
                    # Parsear JSON
                    input_json = json.loads(user_input)

                    # Convertir a DataFrame
                    if isinstance(input_json, list):
                        df_raw = pd.DataFrame(input_json)
                    elif isinstance(input_json, dict):
                        df_raw = pd.DataFrame([input_json])
                    else:
                        st.error("❌ Formato no soportado")
                        st.stop()

                    st.success(f"✅ JSON válido: **{len(df_raw)} producto(s)**")

                    # Preprocesar y predecir
                    df_processed = preprocessor.transform(df_raw.copy())
                    if 'condition' in df_processed.columns:
                        df_processed = df_processed.drop(columns=["condition"])

                    df_result = add_predictions_with_probs(df_raw, df_processed, model)

                    # Mostrar resultados (función estandarizada)
                    display_results(df_result, show_distribution_chart=True)

                    # Descargar
                    json_output = df_result.to_json(orient="records", indent=2)
                    st.download_button(
                        "⬇️ Descargar resultado JSON",
                        json_output,
                        "prediccion.json",
                        "application/json",
                        use_container_width=True
                    )

            except json.JSONDecodeError as e:
                st.error(f"❌ JSON inválido: {e}")
            except Exception as e:
                st.error(f"❌ Error al procesar: {e}")
                with st.expander("Ver detalles del error"):
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 Clasificador MercadoLibre | Modelo: XGBoost | Accuracy: ~86%</p>
</div>
""", unsafe_allow_html=True)

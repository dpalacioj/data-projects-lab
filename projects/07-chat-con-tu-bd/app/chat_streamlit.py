"""Chat Streamlit: haz preguntas en espanol a tu base de datos.

Correr desde la carpeta del proyecto:
    uv run streamlit run app/chat_streamlit.py

O con make:
    make serve
"""
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chat_bd.agent_graph import ejecutar as ejecutar_agente  # noqa: E402
from chat_bd.chain_simple import ejecutar as ejecutar_chain  # noqa: E402
from chat_bd.config import DATABASE_URL, LLM_PROVIDER, OLLAMA_MODEL  # noqa: E402
from chat_bd.sql_tools import _url_oculta, schema_como_texto  # noqa: E402

st.set_page_config(page_title="Chat con tu BD", page_icon=":speech_balloon:", layout="wide")

st.title("Chat con tu base de datos")
st.caption(
    "Haz preguntas en espanol y el LLM las convierte a SQL contra la BD cargada "
    "por el proyecto 06."
)

# ---------------------------------------------------------------------------
# Sidebar: configuracion visible
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Configuracion")
    st.code(f"proveedor: {LLM_PROVIDER}", language="text")
    if LLM_PROVIDER == "ollama":
        st.code(f"modelo:    {OLLAMA_MODEL}", language="text")
    st.code(f"BD:        {_url_oculta(DATABASE_URL)}", language="text")

    modo = st.radio(
        "Motor",
        options=["chain (LangChain)", "graph (LangGraph con reintentos)"],
        index=1,
        help=(
            "chain: una sola pasada NL -> SQL -> respuesta.\n\n"
            "graph: si el SQL falla, el agente lee el error y reintenta "
            "hasta 3 veces."
        ),
    )

    with st.expander("Ver schema de la BD"):
        try:
            st.code(schema_como_texto(), language="text")
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer la BD: {exc}")

# ---------------------------------------------------------------------------
# Historial de chat
# ---------------------------------------------------------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])
        if m.get("sql"):
            with st.expander("Ver SQL generado"):
                st.code(m["sql"], language="sql")
        if m.get("resultado") is not None and len(m["resultado"]) > 0:
            with st.expander(f"Ver resultado ({len(m['resultado'])} filas)"):
                st.dataframe(m["resultado"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
pregunta = st.chat_input("Escribe tu pregunta... (ej: 'top 5 productos por ingresos')")

if pregunta:
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            t0 = time.time()
            try:
                if modo.startswith("chain"):
                    r = ejecutar_chain(pregunta)
                    respuesta_txt = r.respuesta
                    sql = r.sql
                    df = r.resultado
                    meta = f"chain · {time.time() - t0:.1f}s"
                else:
                    r = ejecutar_agente(pregunta)
                    respuesta_txt = r.respuesta
                    sql = r.sql
                    df = r.resultado
                    exito = "ok" if r.exito else "giveup"
                    meta = f"graph · {r.intentos} intento(s) · {exito} · {time.time() - t0:.1f}s"
            except Exception as exc:  # noqa: BLE001
                respuesta_txt = f"Error: {exc}"
                sql = None
                df = None
                meta = "error"

        st.markdown(respuesta_txt)
        st.caption(meta)
        if sql:
            with st.expander("Ver SQL generado"):
                st.code(sql, language="sql")
        if df is not None and len(df) > 0:
            with st.expander(f"Ver resultado ({len(df)} filas)"):
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.session_state.mensajes.append(
        {"rol": "assistant", "texto": respuesta_txt, "sql": sql, "resultado": df}
    )

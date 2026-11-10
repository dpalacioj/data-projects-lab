"""Paquete `chat_bd`: chat NL -> SQL con LangChain y LangGraph.

Mapa de modulos:

- `config.py`      proveedor de LLM, modelo, URL de BD, parametros.
- `llm.py`         factory: devuelve el ChatModel segun el proveedor configurado.
- `sql_tools.py`   schema inspector + ejecucion segura (solo lectura).
- `prompts.py`     plantillas de prompt centralizadas.
- `chain_simple.py` LangChain: cadena NL -> SQL -> ejecuta -> responde.
- `agent_graph.py` LangGraph: grafo con auto-correccion de SQL.
"""
__version__ = "0.1.0"

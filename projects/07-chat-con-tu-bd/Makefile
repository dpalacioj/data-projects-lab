# Makefile del proyecto 07-chat-con-tu-bd.
# Rutas relativas a esta carpeta. `make` sin argumentos muestra la ayuda.

.DEFAULT_GOAL := help
.PHONY: help setup env check-ollama check-db pull-model chat-chain chat-graph serve test lint clean

SRC_PATH := projects/07-chat-con-tu-bd/src
PY := PYTHONPATH=$(SRC_PATH) uv run python

help:  ## Lista los comandos disponibles.
	@echo "Comandos del proyecto 07-chat-con-tu-bd:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup:  ## Instala dependencias (extra chat-bd + ingesta-eventos + dev).
	cd ../.. && uv sync --extra chat-bd --extra ingesta-eventos --extra dev

env:  ## Copia .env.example a .env si no existe.
	@if [ ! -f .env ]; then cp .env.example .env && echo "creado .env (editalo si quieres cambiar proveedor/modelo)"; else echo ".env ya existe"; fi

check-ollama:  ## Verifica que Ollama este corriendo y responda.
	@curl -s http://localhost:11434/api/tags > /dev/null 2>&1 \
		&& echo "Ollama responde en localhost:11434" \
		&& curl -s http://localhost:11434/api/tags | python -m json.tool \
		|| echo "Ollama NO responde. Instala desde https://ollama.com y corre: ollama serve"

pull-model:  ## Descarga el modelo por defecto (qwen2.5:7b, ~4GB).
	@command -v ollama >/dev/null 2>&1 \
		|| (echo "Ollama no instalado. Instala desde https://ollama.com"; exit 1)
	ollama pull qwen2.5:7b

check-db:  ## Verifica que la BD del proyecto 06 este cargada.
	cd ../.. && $(PY) -c "from chat_bd.sql_tools import resumen_bd; import json; print(json.dumps(resumen_bd(), indent=2))"

chat-chain:  ## Pregunta al chain simple desde terminal (usa Ollama).
	@read -p "Pregunta: " q; \
	cd ../.. && $(PY) -c "from chat_bd.chain_simple import responder; print(responder('$$q'))"

chat-graph:  ## Pregunta al agente LangGraph con auto-correccion.
	@read -p "Pregunta: " q; \
	cd ../.. && $(PY) -c "from chat_bd.agent_graph import responder; print(responder('$$q'))"

serve:  ## Lanza el chat Streamlit en http://localhost:8501.
	cd ../.. && PYTHONPATH=$(SRC_PATH) uv run streamlit run projects/07-chat-con-tu-bd/app/chat_streamlit.py

test:  ## Corre los tests (usan LLM falso, no necesitan Ollama).
	cd ../.. && uv run pytest projects/07-chat-con-tu-bd/tests/

lint:  ## Verifica estilo con ruff.
	cd ../.. && uv run ruff check projects/07-chat-con-tu-bd/src/ projects/07-chat-con-tu-bd/tests/

clean:  ## Borra caches locales.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

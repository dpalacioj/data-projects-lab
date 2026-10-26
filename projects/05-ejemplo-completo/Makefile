# Makefile propio del proyecto 05-ejemplo-completo.
# Las rutas son relativas a esta carpeta. Corre `make` sin argumentos para ver la ayuda.

.DEFAULT_GOAL := help
.PHONY: help setup train test lint serve-streamlit serve-api clean

help:  ## Lista los comandos disponibles.
	@echo "Comandos del proyecto 05-ejemplo-completo:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup:  ## Instala las dependencias de este proyecto (extra ejemplo-completo + dev).
	cd ../.. && uv sync --extra ejemplo-completo --extra dev

train:  ## Entrena el modelo y lo guarda en models/clasificador_vino.joblib.
	cd ../.. && uv run python -m ejemplo_completo.train

test:  ## Corre los tests del proyecto.
	cd ../.. && uv run pytest projects/05-ejemplo-completo/tests/

lint:  ## Verifica estilo con ruff.
	cd ../.. && uv run ruff check projects/05-ejemplo-completo/src/ projects/05-ejemplo-completo/tests/

serve-streamlit:  ## Lanza la app Streamlit en http://localhost:8501.
	cd ../.. && uv run streamlit run projects/05-ejemplo-completo/app/streamlit_app.py

serve-api:  ## Lanza la API FastAPI en http://localhost:8000.
	cd ../.. && uv run uvicorn app.api_fastapi:app --reload --app-dir projects/05-ejemplo-completo

clean:  ## Borra caches locales y el modelo entrenado.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -f models/clasificador_vino.joblib

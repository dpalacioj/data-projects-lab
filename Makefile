# Interfaz unica de comandos del repositorio.
# `make` sin argumentos muestra esta ayuda.
#
# Si este archivo te parece denso (que son @, .PHONY, $(MAKEFILE_LIST)...),
# lee tutorials/03-el-makefile-del-curso.md: explica cada linea.

.DEFAULT_GOAL := help
.PHONY: help setup setup-caso smoke test test-slow test-all lint clean app-recompra api-recompra train-recompra app-segmentacion features-segmentacion preparar-datos descargar-datos

help:  ## Lista los comandos disponibles.
	@echo "Comandos disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup:  ## Instala base del curso + grupo dev (equivale a `uv sync --extra dev`).
	uv sync --extra dev

setup-caso:  ## Instala lo necesario para el caso guia (S02/S03): segmentacion + dev.
	uv sync --extra segmentacion-clientes --extra dev

smoke:  ## Diagnostica el entorno (no requiere dependencias instaladas).
	python3 scripts/smoke_test.py

test:  ## Corre los tests rapidos (excluye los marcados -m slow).
	uv run pytest -m "not slow"

test-slow:  ## Corre los tests slow (requieren data/raw y models/recompra_v1/ reentrenado).
	uv run pytest -m slow

test-all:  ## Corre absolutamente todos los tests (rapidos + slow).
	uv run pytest

lint:  ## Corre ruff sobre el codigo del caso guia.
	uv run ruff check projects/00-segmentacion-clientes/

clean:  ## Borra caches de python y de pytest.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

app-recompra:  ## Levanta la app de Streamlit del caso guia (http://localhost:8501).
	cd projects/00-segmentacion-clientes && uv run streamlit run app/recompra_streamlit.py

api-recompra:  ## Levanta la API FastAPI del caso guia (http://127.0.0.1:8000/docs).
	cd projects/00-segmentacion-clientes && uv run uvicorn api.main:app --reload

train-recompra:  ## Entrena models/recompra_v1/ del caso guia (requiere el parquet descargado).
	cd projects/00-segmentacion-clientes && uv run python scripts/entrenar_recompra.py

descargar-datos:  ## Descarga Online Retail II y lo deja en data/raw/online_retail_II.parquet.
	cd projects/00-segmentacion-clientes && uv run python scripts/descargar_datos.py

preparar-datos:  ## Limpia el raw y escribe data/processed/{ventas_validas,cancelaciones}.parquet.
	cd projects/00-segmentacion-clientes && uv run python scripts/preparar_datos.py

features-segmentacion:  ## Genera data/processed/features_clientes.parquet (requiere `make preparar-datos`).
	cd projects/00-segmentacion-clientes && uv run python scripts/generar_features_clientes.py

app-segmentacion:  ## Levanta la app de Streamlit no supervisada (http://localhost:8501).
	cd projects/00-segmentacion-clientes && uv run streamlit run app/streamlit_app.py

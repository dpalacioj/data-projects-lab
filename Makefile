# Interfaz unica de comandos del repositorio.
# `make` sin argumentos muestra esta ayuda.
#
# Si este archivo te parece denso (que son @, .PHONY, $(MAKEFILE_LIST)...),
# lee tutorials/02c-el-makefile-del-curso.md: explica cada linea.

.DEFAULT_GOAL := help
.PHONY: help setup setup-caso smoke test lint clean

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

test:  ## Corre los tests rapidos del caso guia.
	uv run pytest

test-slow:  ## Corre los tests que requieren los datos reales descargados.
	uv run pytest -m slow

lint:  ## Corre ruff sobre el codigo del caso guia.
	uv run ruff check projects/segmentacion-clientes/

clean:  ## Borra caches de python y de pytest.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

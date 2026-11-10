# Makefile del proyecto 06-ingesta-eventos-ecommerce.
# Las rutas son relativas a esta carpeta. Corre `make` sin argumentos para ver la ayuda.

.DEFAULT_GOAL := help
.PHONY: help setup env generate-data generate-data-small check-db init-db load pipeline query serve test lint clean clean-db

# PYTHONPATH apunta a src/ para que `python -m ingesta_eventos.*` funcione
# sin tener que instalar el paquete en el venv (pytest ya lo resuelve por su
# cuenta via el `pythonpath` del pyproject.toml).
SRC_PATH := projects/06-ingesta-eventos-ecommerce/src
PY := PYTHONPATH=$(SRC_PATH) uv run python

help:  ## Lista los comandos disponibles.
	@echo "Comandos del proyecto 06-ingesta-eventos-ecommerce:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup:  ## Instala las dependencias del proyecto (extra ingesta-eventos + dev).
	cd ../.. && uv sync --extra ingesta-eventos --extra dev

env:  ## Copia .env.example a .env si todavia no existe.
	@if [ ! -f .env ]; then cp .env.example .env && echo "creado .env (editalo si quieres cambiar credenciales)"; else echo ".env ya existe"; fi

generate-data:  ## Genera los JSON sinteticos con valores por defecto (~1M+ registros).
	cd ../.. && $(PY) projects/06-ingesta-eventos-ecommerce/scripts/generar_eventos.py

generate-data-small:  ## Genera una muestra pequena (7 dias) para pruebas rapidas.
	cd ../.. && $(PY) projects/06-ingesta-eventos-ecommerce/scripts/generar_eventos.py --dias 7 --pedidos-por-dia 300

check-db:  ## Verifica que la conexion a la base funcione.
	cd ../.. && $(PY) -c "from ingesta_eventos.db import crear_engine, probar_conexion; import json; print(json.dumps(probar_conexion(crear_engine()), indent=2))"

init-db:  ## Crea las tablas vacias (DDL) sin cargar datos.
	cd ../.. && $(PY) -c "from ingesta_eventos.db import crear_engine, crear_tablas; crear_tablas(crear_engine()); print('tablas creadas')"

load:  ## Ejecuta el pipeline ETL: lee JSON -> aplana -> carga a la base.
	cd ../.. && $(PY) -m ingesta_eventos.pipeline

pipeline: generate-data load  ## Genera datos y los carga de un solo tiron.

query:  ## Muestra conteos y queries de ejemplo contra la base cargada.
	cd ../.. && $(PY) -c "from ingesta_eventos.db import crear_engine; from ingesta_eventos.queries import conteos_tablas, ingresos_por_canal, top_productos; e = crear_engine(); import json; print('conteos:', json.dumps(conteos_tablas(e), indent=2)); print(); print('ingresos por canal:'); print(ingresos_por_canal(e)); print(); print('top 5 productos:'); print(top_productos(e, 5))"

serve:  ## Lanza el dashboard Streamlit en http://localhost:8501.
	cd ../.. && PYTHONPATH=$(SRC_PATH) uv run streamlit run projects/06-ingesta-eventos-ecommerce/app/streamlit_app.py

test:  ## Corre los tests del proyecto.
	cd ../.. && uv run pytest projects/06-ingesta-eventos-ecommerce/tests/

lint:  ## Verifica estilo con ruff.
	cd ../.. && uv run ruff check projects/06-ingesta-eventos-ecommerce/src/ projects/06-ingesta-eventos-ecommerce/tests/ projects/06-ingesta-eventos-ecommerce/scripts/

clean:  ## Borra caches locales. NO toca la base ni los JSON.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-db:  ## Borra el archivo SQLite local (empezar desde cero).
	rm -f data/db/eventos.db data/db/eventos.db-*
	@echo "base borrada; corre 'make load' para recargar"

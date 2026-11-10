"""Paquete `ingesta_eventos`: ETL de eventos JSON anidados a una base relacional.

Mapa rapido de modulos (una pregunta -> un archivo):

- `config.py`    donde estan las cosas (rutas, URL de BD, parametros).
- `schemas.py`   que forma tiene cada JSON (modelos Pydantic).
- `db.py`        como me conecto a la base (engine, sesion, DDL).
- `extract.py`   como leo los JSON del disco.
- `transform.py` como aplano el JSON a 3 tablas normalizadas.
- `load.py`      como escribo esas tablas en la base.
- `queries.py`   que preguntas de negocio le hago a la base.
- `pipeline.py`  orquesta extract -> transform -> load.
"""
__version__ = "0.1.0"

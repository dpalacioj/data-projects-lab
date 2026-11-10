"""Orquestador: extract -> transform -> load, y punto de entrada CLI.

Ejecutable como:

    uv run python -m ingesta_eventos.pipeline
    uv run python -m ingesta_eventos.pipeline --reset   # tira y recrea tablas

El comando reporta progreso por lote y un resumen al final.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sqlalchemy.engine import Engine

from .config import RAW_DIR, database_url_safe
from .db import borrar_tablas, crear_engine, crear_tablas, probar_conexion
from .extract import leer_lote, listar_lotes
from .load import cargar_lote
from .queries import conteos_tablas
from .transform import aplanar_lote


def ejecutar(
    engine: Engine | None = None,
    directorio: Path = RAW_DIR,
    reset: bool = False,
    verbose: bool = True,
) -> dict:
    """Corre el pipeline completo sobre todos los JSON de `directorio`.

    Devuelve un dict con los totales cargados (para tests y reporting).
    """
    engine = engine or crear_engine()

    if verbose:
        info = probar_conexion(engine)
        print(f"conectado a: {info['url']}  (driver: {info['driver']})")

    if reset:
        if verbose:
            print("--reset: borrando tablas existentes...")
        borrar_tablas(engine)

    crear_tablas(engine)

    lotes = listar_lotes(directorio)
    if not lotes:
        raise FileNotFoundError(
            f"No se encontraron archivos JSON en {directorio}. "
            "Corre `make generate-data` primero."
        )

    if verbose:
        print(f"lotes a procesar: {len(lotes)}")

    t0 = time.time()
    for i, path in enumerate(lotes, start=1):
        lote = leer_lote(path)
        filas = aplanar_lote(lote)
        conteo = cargar_lote(engine, filas)
        if verbose and (i == 1 or i % 10 == 0 or i == len(lotes)):
            print(
                f"[{i:>3}/{len(lotes)}] {path.name}: "
                f"pedidos={conteo['pedidos']:>5}  items={conteo['items']:>6}"
            )

    # Reportamos el estado REAL de la base al final (COUNT(*) por tabla), no la
    # suma acumulada de filas enviadas: clientes y productos se deduplican
    # entre lotes, asi que la suma de envios sobreestima.
    total = conteos_tablas(engine)
    # Renombramos la clave para que el resumen case con el resto del codigo.
    total["items"] = total.pop("items_pedido")
    dur = time.time() - t0

    if verbose:
        print()
        print("resumen de carga (filas en la base):")
        for k, v in total.items():
            print(f"  {k:<12} {v:>10,}")
        print(f"  {'TOTAL FILAS':<12} {sum(total.values()):>10,}")
        print(f"tiempo total: {dur:.1f}s")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de ingesta: JSON -> base relacional."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra las tablas antes de cargar (idempotencia garantizada).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=RAW_DIR,
        help="Directorio con los JSON de entrada (default: data/raw/).",
    )
    args = parser.parse_args()

    print(f"URL de la base: {database_url_safe()}")
    ejecutar(directorio=args.dir, reset=args.reset)


if __name__ == "__main__":
    main()

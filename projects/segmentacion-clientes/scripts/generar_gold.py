# -*- coding: utf-8 -*-
"""Genera `tests/fixtures/categorias_gold.csv` con 200 descripciones a etiquetar.

El fichero se usa como linea base del cuaderno 08 (`llm_como_transformador`)
para comparar la exactitud del LLM contra una regla por palabras clave.
Las etiquetas (`categoria`) se completan **a mano** por David Palacio;
este script solo deja el CSV con la columna vacia.

Uso:
    uv run python scripts/generar_gold.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segmentacion.data.cargar import leer_parquet  # noqa: E402
from segmentacion.data.limpieza import preparar_ventas  # noqa: E402

PARQUET = PROJECT_ROOT / "data" / "raw" / "online_retail_II.parquet"
SALIDA = PROJECT_ROOT / "tests" / "fixtures" / "categorias_gold.csv"
SEMILLA = 42
N = 200


def main() -> int:
    if not PARQUET.exists():
        print(f"no existe {PARQUET}; ejecuta antes `scripts/descargar_datos.py`.")
        return 2

    df = leer_parquet(PARQUET)
    ventas, _ = preparar_ventas(df, log=False)

    descripciones = (
        ventas.select("Description")
        .drop_nulls()
        .unique()
        .sort("Description")  # orden estable antes del shuffle
    )
    muestra = descripciones.sample(n=N, seed=SEMILLA).sort("Description")
    print(f"muestra: {muestra.height} descripciones unicas (seed={SEMILLA})")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Description", "categoria"])
        for d in muestra["Description"].to_list():
            w.writerow([d, ""])
    print(f"CSV escrito en: {SALIDA}")
    print("Completa la columna `categoria` a mano con una de las 12 etiquetas:")
    print("  iluminacion, cocina, textil, papeleria, decoracion, bolsas,")
    print("  juguetes, jardin, navidad, accesorios, cargo_no_producto, otro")
    return 0


if __name__ == "__main__":
    sys.exit(main())

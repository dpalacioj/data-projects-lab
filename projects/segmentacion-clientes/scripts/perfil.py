# -*- coding: utf-8 -*-
"""Perfil HTML de la muestra de ventas validas (resumen del cuaderno 05).

Uso:
    uv run python scripts/perfil.py

Entrada:  `data/processed/ventas_validas.parquet` (lo deja el cuaderno 02).
Salida :  `reports/perfil_muestra.html` (gitignored).

Pensado para CI / grabacion: genera el HTML sin abrir Jupyter.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_profiling import ProfileReport  # noqa: E402
from segmentacion.data.cargar import leer_parquet  # noqa: E402

VENTAS = PROJECT_ROOT / "data" / "processed" / "ventas_validas.parquet"
SALIDA = PROJECT_ROOT / "reports" / "perfil_muestra.html"
SEMILLA = 42
TAMANIO_MUESTRA = 100_000


def main() -> int:
    if not VENTAS.exists():
        print(f"no existe {VENTAS}; ejecuta antes el cuaderno 02_limpieza.ipynb.")
        return 2

    ventas = leer_parquet(VENTAS)
    print(f"ventas: {ventas.height:,}".replace(",", " "))

    muestra = ventas.sample(n=TAMANIO_MUESTRA, seed=SEMILLA).to_pandas()
    print(f"muestra: {muestra.shape[0]:,} filas (seed={SEMILLA}).".replace(",", " "))

    perfil = ProfileReport(
        muestra,
        title="Perfil muestra — Online Retail II (U2)",
        minimal=True,
        progress_bar=False,
    )
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    perfil.to_file(SALIDA)
    print(f"html escrito en: {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

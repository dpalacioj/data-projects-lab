"""Fixtures compartidos de los tests del proyecto segmentacion-clientes."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_REAL = PROJECT_ROOT / "data" / "raw" / "online_retail_II.parquet"


@pytest.fixture(scope="session")
def parquet_real() -> Path:
    """Ruta al Parquet real. Omite el test si el archivo no existe aun.

    Se usa en los tests marcados `slow` que reproducen la seccion 4 de la spec.
    Para generar el archivo: `uv run python scripts/descargar_datos.py`.
    """
    if not PARQUET_REAL.exists():
        pytest.skip(
            "data/raw/online_retail_II.parquet no existe; "
            "ejecuta `uv run python scripts/descargar_datos.py`."
        )
    return PARQUET_REAL

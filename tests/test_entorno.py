"""Smoke test del repositorio ejecutable desde pytest.

No duplica logica de `scripts/smoke_test.py`: solo verifica que las piezas
estructurales que las sesiones **asumen** existentes siguen ahi. Si este test
falla, hay un cambio de estructura que debe reflejarse en los README de
sesion antes de mergear.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def test_estructura_minima_existe() -> None:
    rutas_requeridas = [
        "sesiones/README.md",
        "sesiones/s01-objetivo-crisp-dm/README.md",
        "sesiones/s02-exploracion-polars/README.md",
        "sesiones/s03-procesamiento-sklearn/README.md",
        "sesiones/s04-app-mlops/README.md",
        "projects/segmentacion-clientes/README.md",
        "docs/specs/SPEC_segmentacion-clientes_polars.md",
        "scripts/smoke_test.py",
        "pyproject.toml",
        "Makefile",
    ]
    faltantes = [r for r in rutas_requeridas if not (RAIZ / r).exists()]
    assert not faltantes, f"Rutas que las sesiones asumen presentes y no estan: {faltantes}"


def test_smoke_script_corre_sin_dependencias() -> None:
    resultado = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "smoke_test.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode in (0, 1), (
        f"smoke_test.py retorno codigo inesperado {resultado.returncode}\n"
        f"stdout: {resultado.stdout}\nstderr: {resultado.stderr}"
    )
    assert "Diagnostico del entorno" in resultado.stdout

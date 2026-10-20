"""Diagnostico del entorno para `data-projects-lab`.

Corre con cualquier Python 3.11+ sin dependencias externas. No importa el
paquete del curso: su objetivo es responder "esta maquina esta lista para
seguir las sesiones?" **antes** de que el estudiante instale nada.

Salida: una tabla con OK / WARN / FAIL por verificacion. Cada FAIL trae el
comando que lo arregla. Si todo sale OK, estas listo para `make setup`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

RAIZ = Path(__file__).resolve().parent.parent

VERDE = "\033[32m"
AMARILLO = "\033[33m"
ROJO = "\033[31m"
RESET = "\033[0m"


def pintar(estado: str) -> str:
    color = {"OK": VERDE, "WARN": AMARILLO, "FAIL": ROJO}[estado]
    return f"{color}{estado:4}{RESET}"


def imprimir_fila(nombre: str, estado: str, detalle: str, remedio: Optional[str]) -> None:
    print(f"  [{pintar(estado)}]  {nombre:30}  {detalle}")
    if remedio and estado != "OK":
        print(f"            ↳ {remedio}")


# --- Verificaciones ---------------------------------------------------------


def check_python() -> tuple[str, str, Optional[str]]:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 11:
        return "OK", version, None
    return (
        "FAIL",
        f"{version} (se necesita 3.11+)",
        "Instala Python 3.11 con uv: `uv python install 3.11`",
    )


def check_uv() -> tuple[str, str, Optional[str]]:
    ruta = shutil.which("uv")
    if ruta is None:
        return (
            "FAIL",
            "no encontrado",
            "macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
    try:
        salida = subprocess.run(
            [ruta, "--version"], capture_output=True, text=True, timeout=5
        )
        return "OK", salida.stdout.strip(), None
    except (subprocess.SubprocessError, OSError):
        return "WARN", "encontrado pero no responde", "Reinstala uv desde astral.sh/uv"


def check_git() -> tuple[str, str, Optional[str]]:
    ruta = shutil.which("git")
    if ruta is None:
        return "FAIL", "no encontrado", "Instala git desde https://git-scm.com/downloads"
    try:
        salida = subprocess.run(
            [ruta, "--version"], capture_output=True, text=True, timeout=5
        )
        return "OK", salida.stdout.strip().replace("git version ", ""), None
    except (subprocess.SubprocessError, OSError):
        return "WARN", "encontrado pero no responde", None


def check_estructura() -> tuple[str, str, Optional[str]]:
    faltantes = [
        d
        for d in ("sesiones", "projects", "docs", "pyproject.toml")
        if not (RAIZ / d).exists()
    ]
    if not faltantes:
        return "OK", "sesiones/ projects/ docs/ pyproject.toml", None
    return (
        "FAIL",
        f"faltan: {', '.join(faltantes)}",
        "Clona el repositorio completo (no copies archivos sueltos)",
    )


def check_pyproject() -> tuple[str, str, Optional[str]]:
    archivo = RAIZ / "pyproject.toml"
    if not archivo.exists():
        return "FAIL", "no existe", "Asegurate de estar en la raiz del repositorio"
    texto = archivo.read_text(encoding="utf-8")
    extras = [n for n in ("segmentacion-clientes", "titanic", "dev") if f"{n} =" in texto or f'"{n}"' in texto]
    if len(extras) >= 2:
        return "OK", f"extras detectados: {len(extras)}", None
    return "WARN", "pyproject.toml existe pero sin extras esperados", None


def check_venv() -> tuple[str, str, Optional[str]]:
    venv = RAIZ / ".venv"
    if venv.exists() and venv.is_dir():
        return "OK", ".venv/ presente", None
    return (
        "WARN",
        ".venv/ no existe todavia",
        "Corre `make setup` (o `uv sync --extra dev`) para crear el ambiente",
    )


def check_dependencia_base(nombre_modulo: str) -> Callable[[], tuple[str, str, Optional[str]]]:
    def _check() -> tuple[str, str, Optional[str]]:
        venv_py = RAIZ / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
        interprete = str(venv_py) if venv_py.exists() else sys.executable
        try:
            r = subprocess.run(
                [interprete, "-c", f"import {nombre_modulo}; print({nombre_modulo}.__version__)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return "OK", r.stdout.strip(), None
            return (
                "WARN",
                "no instalado en el ambiente",
                "Corre `make setup` (o `uv sync --extra dev`) y vuelve a probar",
            )
        except (subprocess.SubprocessError, OSError) as e:
            return "WARN", f"error al verificar: {e}", None

    return _check


# --- Ejecucion --------------------------------------------------------------


VERIFICACIONES: list[tuple[str, Callable[[], tuple[str, str, Optional[str]]]]] = [
    ("Python 3.11+", check_python),
    ("uv", check_uv),
    ("git", check_git),
    ("Estructura del repo", check_estructura),
    ("pyproject.toml", check_pyproject),
    (".venv/", check_venv),
    ("pandas", check_dependencia_base("pandas")),
    ("scikit-learn", check_dependencia_base("sklearn")),
    ("jupyter (core)", check_dependencia_base("jupyter_core")),
]


def main() -> int:
    print()
    print("  Diagnostico del entorno · data-projects-lab")
    print("  " + "-" * 70)
    estados: list[str] = []
    for nombre, fn in VERIFICACIONES:
        estado, detalle, remedio = fn()
        imprimir_fila(nombre, estado, detalle, remedio)
        estados.append(estado)
    print()
    fails = estados.count("FAIL")
    warns = estados.count("WARN")
    if fails:
        print(f"  {ROJO}{fails} verificacion(es) en FAIL.{RESET} Resuelvelas antes de seguir.")
        return 1
    if warns:
        print(f"  {AMARILLO}{warns} advertencia(s).{RESET} Puedes continuar; corre `make setup` si no lo has hecho.")
        return 0
    print(f"  {VERDE}Todo listo.{RESET} Puedes abrir el material en sesiones/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

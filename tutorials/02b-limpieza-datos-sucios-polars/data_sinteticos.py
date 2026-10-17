# -*- coding: utf-8 -*-
"""Genera `datos_sucios.csv` para el tutorial 02b (limpieza en Polars).

Dataset sintetico: 200 pedidos de un e-commerce ficticio con problemas
inventados a mano (sin copiar nada de datasets externos):

- `ciudad`: mismas ciudades escritas de 3 formas distintas (mayusculas,
  minusculas, con tildes/sin tildes).
- `fecha`: 50 % en formato ISO (`2026-10-16`), 50 % en formato europeo
  (`16/10/2026`).
- `precio`: la mitad en EUR, la otra marcada con prefijo `USD:`.
- `cantidad`: valores centinela `-999` indicando "faltante".
- `email`: nulos marcados como `"unknown"` y `None`.
- `producto_id`: duplicados exactos (errores de carga) + duplicados
  "cercanos" (mismo pedido con `cantidad` distinta por una devolucion).

Uso:
    uv run python data_sinteticos.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from random import Random

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "datos_sucios.csv"
SEMILLA = 42

CIUDADES_CANON = ["Barcelona", "Lisboa", "Madrid", "Paris", "Londres"]
VARIANTES_CIUDAD = {
    "Barcelona": ["Barcelona", "barcelona", "BARCELONA", "Barçelona"],
    "Lisboa": ["Lisboa", "LISBOA", "lisboa", "Lisbôa"],
    "Madrid": ["Madrid", "madrid", "MADRID", "Madríd"],
    "Paris": ["Paris", "PARIS", "Parìs", "paris"],
    "Londres": ["Londres", "LONDRES", "londres", "Lóndres"],
}

PRODUCTOS = [f"P{str(n).zfill(3)}" for n in range(1, 41)]


def main() -> int:
    rng = Random(SEMILLA)
    filas = []
    for i in range(1, 180):
        ciudad_canon = rng.choice(CIUDADES_CANON)
        ciudad = rng.choice(VARIANTES_CIUDAD[ciudad_canon])

        if rng.random() < 0.5:
            fecha = f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}"
        else:
            fecha = f"{rng.randint(1, 28):02d}/{rng.randint(1, 10):02d}/2026"

        precio_raw = round(rng.uniform(5, 300), 2)
        if rng.random() < 0.3:
            precio = f"USD:{precio_raw}"
        else:
            precio = f"{precio_raw}"

        cantidad = -999 if rng.random() < 0.05 else rng.randint(1, 10)

        if rng.random() < 0.08:
            email = "unknown"
        elif rng.random() < 0.05:
            email = ""
        else:
            email = f"cliente{i}@ejemplo.com"

        producto = rng.choice(PRODUCTOS)
        filas.append([i, ciudad, fecha, precio, cantidad, email, producto])

    # Duplicados exactos (3) + duplicados cercanos (3)
    for _ in range(3):
        idx = rng.randint(0, len(filas) - 1)
        duplicado = filas[idx].copy()
        duplicado[0] = len(filas) + 1
        filas.append(duplicado)
    for _ in range(3):
        idx = rng.randint(0, len(filas) - 1)
        duplicado = filas[idx].copy()
        duplicado[0] = len(filas) + 1
        duplicado[4] = -duplicado[4] if isinstance(duplicado[4], int) and duplicado[4] > 0 else 1
        filas.append(duplicado)

    # Rellenar hasta 200
    while len(filas) < 200:
        i = len(filas) + 1
        ciudad = rng.choice(VARIANTES_CIUDAD[rng.choice(CIUDADES_CANON)])
        fecha = f"{rng.randint(1, 28):02d}/{rng.randint(1, 10):02d}/2026"
        precio = f"{round(rng.uniform(5, 300), 2)}"
        filas.append([i, ciudad, fecha, precio, 1, f"cliente{i}@ejemplo.com", "P001"])

    rng.shuffle(filas)
    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ciudad", "fecha", "precio", "cantidad", "email", "producto"])
        w.writerows(filas)
    print(f"escrito: {SALIDA} ({len(filas)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

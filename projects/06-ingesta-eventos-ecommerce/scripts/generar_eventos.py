"""Genera archivos JSON sinteticos de pedidos de un e-commerce.

Simula un escenario realista: cada dia la tienda deposita un archivo JSON
con todos los pedidos de ese dia. La estructura es anidada (cada pedido
tiene un cliente y una lista variable de items).

Uso:

    uv run python scripts/generar_eventos.py                    # defaults (.env)
    uv run python scripts/generar_eventos.py --dias 7           # muestra pequena
    uv run python scripts/generar_eventos.py --out data/raw     # otra ruta

Con los defaults (60 dias x 1.700 pedidos/dia, media de 10 items por pedido)
se generan ~102.000 pedidos y ~1.020.000 items. Sumado a 20.000 clientes y
500 productos, el total supera **900.000 registros**.

El generador NO usa librerias pesadas como Faker: trabaja con pools finitos de
nombres, ciudades y categorias que mezcla con `random.choices`. Es 20-50x mas
rapido que Faker para este tamano, lo que permite generar 1M+ registros en
menos de un minuto.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Permite correr el script sin instalar el paquete (patron del repo).
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingesta_eventos.config import (  # noqa: E402
    DIAS,
    N_CLIENTES,
    N_PRODUCTOS,
    PEDIDOS_POR_DIA,
    RAW_DIR,
    SEMILLA,
)

# ---------------------------------------------------------------------------
# Pools de valores sinteticos (pequenos, se recombinan)
# ---------------------------------------------------------------------------
NOMBRES = [
    "Ana", "Juan", "Maria", "Carlos", "Lucia", "Pedro", "Sofia", "Miguel",
    "Laura", "Diego", "Valentina", "Andres", "Camila", "Jose", "Daniela",
    "Felipe", "Natalia", "Santiago", "Isabella", "Mateo", "Gabriela",
    "Sebastian", "Paula", "Tomas", "Valeria", "Nicolas", "Mariana",
    "Alejandro", "Catalina", "Emilio",
]
APELLIDOS = [
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Perez", "Sanchez",
    "Ramirez", "Torres", "Flores", "Rivera", "Gomez", "Diaz", "Cruz", "Morales",
    "Reyes", "Ortiz", "Gutierrez", "Chavez", "Ruiz", "Mendoza", "Herrera",
    "Castillo", "Jimenez", "Vargas", "Castro", "Romero", "Alvarez", "Soto",
    "Silva",
]
CIUDADES_PAISES = [
    ("Medellin", "Colombia"), ("Bogota", "Colombia"), ("Cali", "Colombia"),
    ("Ciudad de Mexico", "Mexico"), ("Guadalajara", "Mexico"),
    ("Monterrey", "Mexico"), ("Buenos Aires", "Argentina"),
    ("Cordoba", "Argentina"), ("Lima", "Peru"), ("Santiago", "Chile"),
    ("Quito", "Ecuador"), ("Montevideo", "Uruguay"), ("Madrid", "Espana"),
    ("Barcelona", "Espana"), ("Sao Paulo", "Brasil"),
]
CATEGORIAS = [
    "Electronica", "Hogar", "Moda", "Deportes", "Libros", "Juguetes",
    "Cocina", "Belleza", "Oficina", "Jardineria",
]
PRODUCTOS_BASE = [
    "Auriculares", "Camiseta", "Zapatos", "Reloj", "Libro", "Lampara",
    "Taza", "Mochila", "Teclado", "Mouse", "Monitor", "Silla",
    "Cafetera", "Tostadora", "Pelota", "Raqueta", "Perfume", "Crema",
    "Mesa", "Cojin", "Puzzle", "Muneca", "Carrito", "Boligrafo",
    "Cuaderno", "Macetero", "Regadera",
]
CANALES = ["web", "app", "marketplace"]
PESOS_CANALES = [0.55, 0.30, 0.15]  # web domina, marketplace es minoritario


def _nombre_completo(rng: random.Random) -> str:
    return f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)}"


def _email_desde_nombre(nombre: str, cid: str) -> str:
    base = nombre.lower().replace(" ", ".")
    return f"{base}.{cid.split('-')[-1]}@mail.test"


def construir_pool_clientes(n: int, rng: random.Random) -> list[dict]:
    """Genera `n` clientes unicos. Se usan como pool para asignar a pedidos."""
    clientes = []
    for i in range(n):
        cid = f"C-{i:06d}"
        nombre = _nombre_completo(rng)
        ciudad, pais = rng.choice(CIUDADES_PAISES)
        clientes.append(
            {
                "cliente_id": cid,
                "nombre": nombre,
                "email": _email_desde_nombre(nombre, cid),
                "ciudad": ciudad,
                "pais": pais,
            }
        )
    return clientes


def construir_pool_productos(n: int, rng: random.Random) -> list[dict]:
    """Catalogo fijo de productos: nombre + categoria + precio base."""
    productos = []
    for i in range(n):
        base = rng.choice(PRODUCTOS_BASE)
        categoria = rng.choice(CATEGORIAS)
        precio = round(rng.uniform(5.0, 400.0), 2)
        productos.append(
            {
                "producto_id": f"SKU-{i:05d}",
                "nombre": f"{base} {categoria} #{i:04d}",
                "categoria": categoria,
                "precio_unitario": precio,
            }
        )
    return productos


def construir_pedido(
    pedido_id: str,
    fecha: datetime,
    cliente: dict,
    productos: list[dict],
    rng: random.Random,
) -> dict:
    """Arma un pedido con 1-25 items (media ~10)."""
    # Distribucion sesgada a la derecha: la mayoria pide pocos, unos pocos muchos.
    n_items = max(1, min(25, int(rng.gauss(mu=10, sigma=5))))
    # `choices` con reemplazo: un cliente puede llevar el mismo producto varias
    # veces (unidades distintas); eso se modela como items separados o cantidad
    # mayor. Aqui simplificamos: productos distintos, cantidades variables.
    if n_items > len(productos):
        n_items = len(productos)
    seleccion = rng.sample(productos, k=n_items)

    items = []
    for prod in seleccion:
        items.append(
            {
                "producto_id": prod["producto_id"],
                "nombre": prod["nombre"],
                "categoria": prod["categoria"],
                "cantidad": rng.randint(1, 4),
                "precio_unitario": prod["precio_unitario"],
            }
        )

    canal = rng.choices(CANALES, weights=PESOS_CANALES, k=1)[0]

    return {
        "pedido_id": pedido_id,
        "fecha": fecha.isoformat(),
        "canal": canal,
        "cliente": {k: v for k, v in cliente.items() if k != "precio_unitario"},
        "items": items,
    }


def generar(
    out_dir: Path = RAW_DIR,
    dias: int = DIAS,
    pedidos_por_dia: int = PEDIDOS_POR_DIA,
    n_clientes: int = N_CLIENTES,
    n_productos: int = N_PRODUCTOS,
    semilla: int = SEMILLA,
    verbose: bool = True,
) -> dict:
    """Genera `dias` archivos JSON, uno por dia. Devuelve conteos totales.

    Nombres de archivo: `eventos_YYYY-MM-DD.json` (ordenables).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(semilla)

    clientes = construir_pool_clientes(n_clientes, rng)
    productos = construir_pool_productos(n_productos, rng)

    # Fecha inicial: hace `dias` dias contados desde hoy (asi los lotes cubren
    # un rango reciente que el estudiante puede filtrar en el dashboard).
    fecha_base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fecha_inicio = fecha_base - timedelta(days=dias)

    total_pedidos = 0
    total_items = 0
    t0 = time.time()

    for d in range(dias):
        dia = fecha_inicio + timedelta(days=d)
        pedidos = []
        for p in range(pedidos_por_dia):
            pedido_id = f"P-{dia.strftime('%Y%m%d')}-{p:05d}"
            # Hora aleatoria dentro del dia.
            fecha = dia + timedelta(
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )
            cliente = rng.choice(clientes)
            pedido = construir_pedido(pedido_id, fecha, cliente, productos, rng)
            pedidos.append(pedido)
            total_items += len(pedido["items"])

        lote = {
            "fecha_lote": dia.strftime("%Y-%m-%d"),
            "version_esquema": "1.0",
            "pedidos": pedidos,
        }

        path = out_dir / f"eventos_{dia.strftime('%Y-%m-%d')}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(lote, f, ensure_ascii=False, separators=(",", ":"))

        total_pedidos += len(pedidos)
        if verbose and (d == 0 or (d + 1) % 10 == 0 or d == dias - 1):
            print(
                f"[{d+1:>3}/{dias}] {path.name}  "
                f"pedidos={len(pedidos):>5}  items_acum={total_items:>8}"
            )

    dur = time.time() - t0
    totales = {
        "clientes": n_clientes,
        "productos": n_productos,
        "pedidos": total_pedidos,
        "items": total_items,
        "total_registros": n_clientes + n_productos + total_pedidos + total_items,
        "segundos": round(dur, 1),
    }
    if verbose:
        print()
        print("resumen:")
        for k, v in totales.items():
            print(f"  {k:<16} {v:>12,}" if isinstance(v, int) else f"  {k:<16} {v:>12}")
    return totales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dias", type=int, default=DIAS)
    parser.add_argument("--pedidos-por-dia", type=int, default=PEDIDOS_POR_DIA)
    parser.add_argument("--clientes", type=int, default=N_CLIENTES)
    parser.add_argument("--productos", type=int, default=N_PRODUCTOS)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    generar(
        out_dir=args.out,
        dias=args.dias,
        pedidos_por_dia=args.pedidos_por_dia,
        n_clientes=args.clientes,
        n_productos=args.productos,
        semilla=args.semilla,
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Descarga Online Retail II (UCI 502) y lo convierte a Parquet.

Uso:
    uv run python scripts/descargar_datos.py

Deja:
    data/raw/online_retail_II.zip
    data/raw/online_retail_II.xlsx
    data/raw/online_retail_II.parquet

Verifica integridad por SHA-256 del zip. Si el hash no coincide, se detiene
y pide actualizar la constante SHA256_ZIP (y el OVA y la spec si procede).

Fuente: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import polars as pl  # noqa: E402
from segmentacion.data.cargar import ESQUEMA, leer_excel_con_esquema  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHA256_ZIP = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "online_retail_II.zip"
XLSX_PATH = RAW_DIR / "online_retail_II.xlsx"
PARQUET_PATH = RAW_DIR / "online_retail_II.parquet"


def descargar(url: str, destino: Path) -> bytes:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        contenido = destino.read_bytes()
        print(f"zip ya existe ({destino.stat().st_size / 1e6:.1f} MB); se reutiliza.")
        return contenido
    print(f"descargando {url} ...")
    req = Request(url, headers={"User-Agent": "data-projects-lab/segmentacion"})
    with urlopen(req, timeout=120) as resp:
        contenido = resp.read()
    destino.write_bytes(contenido)
    print(f"guardado en {destino} ({len(contenido) / 1e6:.1f} MB)")
    return contenido


def verificar_hash(contenido: bytes, esperado: str) -> None:
    observado = hashlib.sha256(contenido).hexdigest()
    print(f"sha256 observado: {observado}")
    if not esperado:
        print("sha256 esperado vacio; actualiza SHA256_ZIP con el valor observado.")
        sys.exit(2)
    if observado != esperado:
        print(f"sha256 no coincide; esperado: {esperado}")
        print("no se continua. verifica si UCI republico el archivo y actualiza OVA y spec.")
        sys.exit(2)
    print("sha256 OK.")


def extraer_xlsx(contenido: bytes, destino: Path) -> None:
    if destino.exists():
        print(f"xlsx ya existe ({destino.stat().st_size / 1e6:.1f} MB); se reutiliza.")
        return
    with zipfile.ZipFile(io.BytesIO(contenido)) as zf:
        nombres = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not nombres:
            raise RuntimeError(f"no se encontro xlsx dentro del zip: {zf.namelist()}")
        with zf.open(nombres[0]) as origen, destino.open("wb") as salida:
            salida.write(origen.read())
    print(f"xlsx extraido a {destino} ({destino.stat().st_size / 1e6:.1f} MB)")


def convertir_a_parquet(xlsx: Path, parquet: Path) -> None:
    if parquet.exists():
        print(f"parquet ya existe ({parquet.stat().st_size / 1e6:.1f} MB); se reutiliza.")
        return
    print(f"leyendo {xlsx} con esquema {ESQUEMA} ...")
    df = leer_excel_con_esquema(xlsx)
    df.write_parquet(parquet, compression="zstd")
    print(
        f"parquet escrito en {parquet}: {df.height:,} filas, {df.width} columnas.".replace(",", " ")
    )


def main() -> int:
    contenido = descargar(URL, ZIP_PATH)
    verificar_hash(contenido, SHA256_ZIP)
    extraer_xlsx(contenido, XLSX_PATH)
    convertir_a_parquet(XLSX_PATH, PARQUET_PATH)
    df = pl.read_parquet(PARQUET_PATH)
    print(
        f"verificacion final: filas={df.height:,}, Invoice.dtype={df.schema['Invoice']}".replace(
            ",", " "
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

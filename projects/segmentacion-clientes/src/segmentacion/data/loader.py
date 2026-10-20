"""
Descarga y carga del dataset *Online Retail II* (UCI ML Repository, ID 502).

El dataset contiene transacciones reales de un retailer británico de regalos
entre 2009-12-01 y 2011-12-09. Es ideal para segmentación de clientes
porque permite construir variables RFM y otras métricas de comportamiento.

Fuente oficial:
    https://archive.ics.uci.edu/dataset/502/online+retail+ii
Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from segmentacion.config import RAW_DATA_DIR, RAW_DATA_FILE, RAW_DATA_URL
from segmentacion.utils.logger import setup_logger

logger = setup_logger(__name__)


def download_dataset(
    url: str = RAW_DATA_URL,
    target_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> Path:
    """
    Descarga el archivo ``online_retail_II.xlsx`` desde el UCI Repository.

    El archivo viene comprimido en un ZIP. Esta función descomprime el
    contenido directamente en ``target_dir``.

    Parameters
    ----------
    url : str
        URL del ZIP en UCI. Por defecto se toma de ``src.config``.
    target_dir : Path
        Carpeta destino para los archivos descomprimidos.
    force : bool
        Si es ``True`` re-descarga aunque el archivo ya exista localmente.

    Returns
    -------
    Path
        Ruta al archivo ``online_retail_II.xlsx`` resultante.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_FILE.exists() and not force:
        logger.info(f"Dataset ya descargado en {RAW_DATA_FILE}. Usa force=True para re-descargar.")
        return RAW_DATA_FILE

    logger.info(f"Descargando dataset desde: {url}")
    with urlopen(url) as response:  # noqa: S310 (URL fija de UCI)
        zip_bytes = response.read()

    logger.info(f"Descarga completa ({len(zip_bytes) / 1e6:.1f} MB). Descomprimiendo...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(target_dir)

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {RAW_DATA_FILE.name} dentro del ZIP. "
            f"Archivos extraídos: {list(target_dir.iterdir())}"
        )

    logger.info(f"Dataset listo en: {RAW_DATA_FILE}")
    return RAW_DATA_FILE


def load_raw_transactions(
    file_path: Path = RAW_DATA_FILE,
    sheets: tuple[str, ...] = ("Year 2009-2010", "Year 2010-2011"),
) -> pd.DataFrame:
    """
    Carga las transacciones desde el archivo Excel y concatena ambas hojas.

    El archivo trae los datos divididos en dos pestañas (un año por hoja).
    Esta función las une en un solo DataFrame y normaliza los nombres
    de columnas a un estándar consistente:

    - ``Invoice``      : número de factura
    - ``StockCode``    : código del producto
    - ``Description``  : descripción del producto
    - ``Quantity``     : cantidad comprada
    - ``InvoiceDate``  : fecha y hora de la factura
    - ``Price``        : precio unitario en libras esterlinas
    - ``CustomerID``   : identificador del cliente (puede ser nulo)
    - ``Country``      : país del cliente

    Parameters
    ----------
    file_path : Path
        Ruta al archivo Excel.
    sheets : tuple[str, ...]
        Pestañas a leer y concatenar.

    Returns
    -------
    pd.DataFrame
        Transacciones combinadas, una fila por línea de factura.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró {file_path}. Ejecuta primero download_dataset().")

    logger.info(f"Leyendo {file_path.name} ({len(sheets)} hojas)...")
    frames = []
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df["__sheet"] = sheet
        frames.append(df)
        logger.info(f"  - '{sheet}': {len(df):,} filas")

    df = pd.concat(frames, ignore_index=True)

    # Normalización de nombres: el archivo de UCI a veces usa 'Customer ID'
    # con espacio. Lo unificamos para evitar errores aguas abajo.
    rename_map = {
        "Customer ID": "CustomerID",
        "Invoice": "Invoice",  # ya es así, pero documentamos la convención
    }
    df = df.rename(columns=rename_map)
    df = df.drop(columns="__sheet")

    logger.info(f"Total transacciones cargadas: {len(df):,}")
    return df

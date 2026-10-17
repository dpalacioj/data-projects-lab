"""Registro de artefactos de modelo para la Unidad 3.

Firma citada por `SPEC_procesamiento_U3.md` §4:

    guardar(pipe, ruta, metadatos)
    cargar(ruta) -> (pipe, meta)

`cargar()` falla con un mensaje claro si la version de scikit-learn del
entorno difiere de la que registro `metadatos.json`. La ficha del artefacto
(`metadatos.json`) es la interfaz publica que consume la Unidad 4 (app
Streamlit): columnas en orden, umbral acordado, hash de los datos.

Diseno de carpeta:

    models/recompra_v1/
    ├── pipeline.joblib     # gitignored
    └── metadatos.json      # versionado

Campos obligatorios de `metadatos` (Tabla 7 del OVA):

- `fecha_corte` (ISO)
- `horizonte` (dias)
- `variables_en_orden` (lista)
- `semilla`
- `versiones` {python, sklearn, polars}
- `datos` {ruta, hash}
- `auc_validacion`
- `umbral_acordado`
- `fecha` (ISO)
- `autor`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

CAMPOS_OBLIGATORIOS = frozenset(
    {
        "fecha_corte",
        "horizonte",
        "variables_en_orden",
        "semilla",
        "versiones",
        "datos",
        "auc_validacion",
        "umbral_acordado",
        "fecha",
        "autor",
    }
)


class MetadatosIncompletos(ValueError):
    """Alguno de los campos de la Tabla 7 falta en `metadatos`."""


class VersionIncompatible(RuntimeError):
    """La version de scikit-learn del entorno difiere de la registrada."""


def _ruta_pipeline(ruta: Path) -> Path:
    return ruta / "pipeline.joblib"


def _ruta_metadatos(ruta: Path) -> Path:
    return ruta / "metadatos.json"


def guardar(pipe: Pipeline, ruta: Path, metadatos: dict[str, Any]) -> None:
    """Serializa `pipe` y `metadatos` en `ruta/{pipeline.joblib, metadatos.json}`.

    Rellena automaticamente `versiones.sklearn` con la version actual si no
    viene incluida. Lanza `MetadatosIncompletos` si falta cualquier campo
    obligatorio.
    """
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)

    metadatos = dict(metadatos)
    versiones = dict(metadatos.get("versiones", {}))
    versiones.setdefault("sklearn", sklearn.__version__)
    metadatos["versiones"] = versiones

    faltantes = CAMPOS_OBLIGATORIOS - set(metadatos.keys())
    if faltantes:
        raise MetadatosIncompletos(
            f"faltan campos en metadatos.json: {sorted(faltantes)} (ver Tabla 7 del OVA U3)."
        )

    joblib.dump(pipe, _ruta_pipeline(ruta))
    _ruta_metadatos(ruta).write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def cargar(ruta: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Lee `pipeline.joblib` y `metadatos.json` desde `ruta`.

    Falla con `VersionIncompatible` si la version de scikit-learn del
    entorno difiere de la registrada en `metadatos["versiones"]["sklearn"]`.
    El mensaje incluye las dos versiones para que el estudiante pueda
    sincronizarlas antes de reintentar.
    """
    ruta = Path(ruta)
    meta_path = _ruta_metadatos(ruta)
    pipe_path = _ruta_pipeline(ruta)
    if not meta_path.exists():
        raise FileNotFoundError(f"no se encontro {meta_path}")
    if not pipe_path.exists():
        raise FileNotFoundError(f"no se encontro {pipe_path}")

    metadatos = json.loads(meta_path.read_text())
    sk_meta = metadatos.get("versiones", {}).get("sklearn")
    sk_env = sklearn.__version__
    if sk_meta is not None and sk_meta != sk_env:
        raise VersionIncompatible(
            "La version de scikit-learn no coincide con la registrada:\n"
            f"  en metadatos.json: {sk_meta}\n"
            f"  en el entorno    : {sk_env}\n"
            "Actualiza el entorno (`uv sync`) o re-entrena con la version actual."
        )

    pipe = joblib.load(pipe_path)
    return pipe, metadatos

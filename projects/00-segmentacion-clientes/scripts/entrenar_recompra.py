# -*- coding: utf-8 -*-
"""Entrena el bosque aleatorio de la Unidad 3 y guarda `models/recompra_v1/`.

Es la version "comando" de la ultima seccion del cuaderno 09: mismas
funciones del paquete (`preparar_ventas`, `cohorte`, `construir_preprocesador`,
`MODELOS`, `entrenar`, `tabla_umbrales`, `guardar`), mismos metadatos de la
Tabla 7 del OVA U3, y dos campos mas que usa la Unidad 4:

- `version_modelo`: nombre de la carpeta (`recompra_v1`).
- `tabla_umbrales`: filas de `tabla_umbrales()` en la cohorte B (Tabla 6 del
  OVA U3), para que la app de Streamlit muestre el efecto del umbral.

Uso (desde la carpeta del proyecto):
    uv run python scripts/entrenar_recompra.py
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings("ignore")

from segmentacion.config import FECHA_CORTE_A, FECHA_CORTE_B, HORIZONTE, SEMILLA  # noqa: E402
from segmentacion.data.cargar import leer_parquet  # noqa: E402
from segmentacion.data.limpieza import preparar_ventas  # noqa: E402
from segmentacion.evaluation.umbral import tabla_umbrales  # noqa: E402
from segmentacion.features.cohortes import cohorte  # noqa: E402
from segmentacion.models.entrenar import MODELOS, entrenar  # noqa: E402
from segmentacion.models.registro import guardar  # noqa: E402
from segmentacion.pipelines.preprocesamiento import (  # noqa: E402
    CAT,
    NUM_LOG,
    NUM_STD,
    construir_preprocesador,
)
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PARQUET_CRUDO = PROJECT_ROOT / "data" / "raw" / "online_retail_II.parquet"
VERSION_MODELO = "recompra_v1"
RUTA = PROJECT_ROOT / "models" / VERSION_MODELO


def main() -> int:
    df = leer_parquet(PARQUET_CRUDO)
    ventas, canc = preparar_ventas(df, log=False)
    A = cohorte(ventas, canc, FECHA_CORTE_A, horizonte=HORIZONTE)
    B = cohorte(ventas, canc, FECHA_CORTE_B, horizonte=HORIZONTE)

    cols = NUM_LOG + NUM_STD + CAT
    XA, yA = A.select(cols).to_pandas(), A["recompra"].to_numpy()
    XB, yB = B.select(cols).to_pandas(), B["recompra"].to_numpy()

    pipe = Pipeline([("pre", construir_preprocesador()), ("mod", MODELOS["Bosque aleatorio"])])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    cv_res = entrenar(pipe, XA, yA, cv=cv)
    pipe.fit(XA, yA)
    proba_B = pipe.predict_proba(XB)[:, 1]
    umbrales = tabla_umbrales(yB, proba_B)

    hash_datos = hashlib.sha256(PARQUET_CRUDO.read_bytes()).hexdigest()[:16]
    metadatos = {
        "fecha_corte": str(FECHA_CORTE_A.date()),
        "horizonte": HORIZONTE,
        "variables_en_orden": cols,
        "semilla": SEMILLA,
        "versiones": {"python": "3.11", "polars": "2.0"},  # sklearn se autocompleta
        "datos": {"ruta": "data/raw/online_retail_II.parquet", "hash": hash_datos},
        "auc_validacion": round(cv_res["cv_auc"], 3),
        "umbral_acordado": 0.3,
        "fecha": datetime.today().date().isoformat(),
        "autor": "David Palacio (U3)",
        # --- campos que agrega la Unidad 4 ---
        "version_modelo": VERSION_MODELO,
        "auc_prueba": round(float(roc_auc_score(yB, proba_B)), 3),
        "tabla_umbrales": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}
            for fila in umbrales.to_dicts()
        ],
    }
    guardar(pipe, RUTA, metadatos)
    print(f"artefacto en {RUTA}/  auc_cv={metadatos['auc_validacion']}  auc_B={metadatos['auc_prueba']}")
    print(umbrales)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Caso supervisado U3: envoltorio del paquete `segmentacion`.

Este script reproduce el caso de recompra llamando a los modulos del
paquete (`features`, `pipelines`, `models`, `evaluation`); su unica
responsabilidad es orquestar y escribir las salidas en `reports/`.

Uso:
    uv run python scripts/caso_recompra.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings("ignore")

import polars as pl  # noqa: E402
from segmentacion.config import FECHA_CORTE_A, FECHA_CORTE_B, HORIZONTE, SEMILLA  # noqa: E402
from segmentacion.data.cargar import leer_parquet  # noqa: E402
from segmentacion.data.limpieza import preparar_ventas  # noqa: E402
from segmentacion.evaluation.metricas import (  # noqa: E402
    curva_profundidad,
    evaluar_clasificacion,
    fuga_escalado,
    fuga_recencia,
    regla_recencia,
)
from segmentacion.evaluation.umbral import tabla_umbrales  # noqa: E402
from segmentacion.features.cohortes import cohorte  # noqa: E402
from segmentacion.models.entrenar import MODELOS, entrenar  # noqa: E402
from segmentacion.pipelines.preprocesamiento import (  # noqa: E402
    CAT,
    NUM_LOG,
    NUM_STD,
    construir_preprocesador,
)
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

PARQUET_CRUDO = PROJECT_ROOT / "data" / "raw" / "online_retail_II.parquet"
REPORTS = PROJECT_ROOT / "reports"


def main() -> int:
    df = leer_parquet(PARQUET_CRUDO)
    ventas, canc = preparar_ventas(df, log=False)
    print(f"ventas {ventas.height}  canc {canc.height}")

    A = cohorte(ventas, canc, FECHA_CORTE_A, horizonte=HORIZONTE)
    B = cohorte(ventas, canc, FECHA_CORTE_B, horizonte=HORIZONTE)

    res: dict = {}
    for nombre, c in (("A", A), ("B", B)):
        res[f"cohorte_{nombre}"] = {
            "n": c.height,
            "pos": int(c["recompra"].sum()),
            "tasa": round(float(c["recompra"].mean()) * 100, 1),
        }

    cols = NUM_LOG + NUM_STD + CAT
    XA, yA = A.select(cols).to_pandas(), A["recompra"].to_numpy()
    XB, yB = B.select(cols).to_pandas(), B["recompra"].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    tabla: dict = {}
    for n, m in MODELOS.items():
        pipe = Pipeline([("pre", construir_preprocesador()), ("mod", m)])
        cv_res = entrenar(pipe, XA, yA, cv=cv)
        pipe.fit(XA, yA)
        proba = pipe.predict_proba(XB)[:, 1]
        test = evaluar_clasificacion(yB, proba, umbral=0.5)
        tabla[n] = {
            "cv_auc": round(cv_res["cv_auc"], 3),
            "cv_auc_sd": round(cv_res["cv_auc_sd"], 3),
            "cv_ap": round(cv_res["cv_ap"], 3),
            "cv_f1": round(cv_res["cv_f1"], 3),
            "test_auc": round(test["auc"], 3),
            "test_ap": round(test["ap"], 3),
            "test_acc": round(test["accuracy"], 3),
            "test_prec": round(test["precision"], 3),
            "test_rec": round(test["recall"], 3),
            "test_f1": round(test["f1"], 3),
        }
        print(n, tabla[n])
    res["modelos"] = tabla

    y_regla = regla_recencia(XB)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    res["regla"] = {
        "acc": round(accuracy_score(yB, y_regla), 3),
        "prec": round(precision_score(yB, y_regla), 3),
        "rec": round(recall_score(yB, y_regla), 3),
        "f1": round(f1_score(yB, y_regla), 3),
    }
    print("base mayoritaria acc", round(max(float(yB.mean()), 1 - float(yB.mean())), 3))
    print("regla recencia<90", res["regla"])

    f_rec = fuga_recencia(
        XA,
        yA,
        XB,
        yB,
        A["recencia_fuga"].to_numpy(),
        B["recencia_fuga"].to_numpy(),
        construir_preprocesador,
    )
    res["fuga_recencia"] = {k: round(v, 3) for k, v in f_rec.items()}
    print("FUGA recencia", res["fuga_recencia"])

    f_esc = fuga_escalado(XA, XB, NUM_STD)
    res["fuga_escalado"] = {
        "media_recencia_todo": round(f_esc["media_recencia_todo"], 1),
        "media_recencia_train": round(f_esc["media_recencia_train"], 1),
    }
    print("media recencia all vs train", res["fuga_escalado"])

    pipe_bosque = Pipeline(
        [("pre", construir_preprocesador()), ("mod", MODELOS["Bosque aleatorio"])]
    ).fit(XA, yA)
    proba_bosque = pipe_bosque.predict_proba(XB)[:, 1]
    um = tabla_umbrales(yB, proba_bosque)
    res["umbrales"] = {
        str(fila["umbral"]): {
            "prec": round(fila["precision"], 3),
            "rec": round(fila["recall"], 3),
            "f1": round(fila["f1"], 3),
            "n_pos": fila["n_senialados"],
        }
        for fila in um.to_dicts()
    }
    print("umbrales", res["umbrales"])

    nombres = pipe_bosque.named_steps["pre"].get_feature_names_out()
    importancias = pipe_bosque.named_steps["mod"].feature_importances_
    imp_s = sorted(zip(nombres, importancias, strict=True), key=lambda x: -x[1])
    res["importancia"] = [(n, round(float(v), 3)) for n, v in imp_s]
    print(imp_s)

    curva = curva_profundidad(XA, yA, XB, yB, construir_preprocesador)
    res["curva_profundidad"] = [(d, round(a, 3), round(b, 3)) for d, a, b in curva]
    print("curva profundidad", res["curva_profundidad"])

    cm = confusion_matrix(yB, (proba_bosque >= 0.5).astype(int))
    res["cm_rf"] = cm.tolist()
    print(cm)

    res["cortes"] = {
        "A": str(FECHA_CORTE_A.date()),
        "B": str(FECHA_CORTE_B.date()),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "resultados.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n"
    )
    A.write_parquet(REPORTS / "cohorte_A.parquet")
    B.with_columns(pl.Series("p_rf", proba_bosque)).write_parquet(REPORTS / "cohorte_B.parquet")

    print(A.select("recencia", "frecuencia", "monetario", "tasa_devolucion").describe())
    print(A.group_by("pais").agg(pl.len(), pl.col("recompra").mean().round(3)))

    # Suprime linter de import de roc_auc_score aunque no lo usemos directamente.
    _ = roc_auc_score
    return 0


if __name__ == "__main__":
    sys.exit(main())

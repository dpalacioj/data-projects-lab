# -*- coding: utf-8 -*-
"""Caso supervisado U3: ¿el cliente vuelve a comprar en los 90 días siguientes a la fecha de corte?"""

import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pl.read_parquet("data/raw/online_retail_II.parquet")
ventas = (
    df.unique()
    .filter(~pl.col("Invoice").str.starts_with("C"))
    .filter(pl.col("Customer ID").is_not_null())
    .filter(pl.col("Quantity") > 0, pl.col("Price") > 0)
    .with_columns((pl.col("Quantity") * pl.col("Price")).alias("Importe"))
)
canc = (
    df.unique()
    .filter(pl.col("Invoice").str.starts_with("C"))
    .filter(pl.col("Customer ID").is_not_null())
)
print("ventas", ventas.height, "canc", canc.height)


def cohorte(corte: datetime, horizonte=90):
    hist = ventas.filter(pl.col("InvoiceDate") < corte)
    fut = ventas.filter(
        (pl.col("InvoiceDate") >= corte)
        & (pl.col("InvoiceDate") < corte + timedelta(days=horizonte))
    )
    chist = canc.filter(pl.col("InvoiceDate") < corte)
    feats = hist.group_by("Customer ID").agg(
        (pl.lit(corte) - pl.col("InvoiceDate").max()).dt.total_days().alias("recencia"),
        (pl.lit(corte) - pl.col("InvoiceDate").min()).dt.total_days().alias("antiguedad"),
        pl.col("Invoice").n_unique().alias("frecuencia"),
        pl.col("Importe").sum().alias("monetario"),
        pl.col("StockCode").n_unique().alias("productos_distintos"),
        pl.col("Country").first().alias("pais"),
    )
    dev = chist.group_by("Customer ID").agg(pl.col("Invoice").n_unique().alias("n_cancelaciones"))
    feats = (
        feats.join(dev, on="Customer ID", how="left")
        .with_columns(
            (pl.col("n_cancelaciones").fill_null(0) / pl.col("frecuencia")).alias(
                "tasa_devolucion"
            ),
            pl.when(pl.col("pais") == "United Kingdom")
            .then(pl.lit("UK"))
            .otherwise(pl.lit("otro"))
            .alias("pais"),
        )
        .drop("n_cancelaciones")
    )
    y = fut.select(pl.col("Customer ID").unique()).with_columns(pl.lit(1).alias("recompra"))
    out = feats.join(y, on="Customer ID", how="left").with_columns(pl.col("recompra").fill_null(0))
    # fuga: recencia calculada con la ultima compra de TODO el periodo (incluye el futuro)
    leak = ventas.group_by("Customer ID").agg(
        (pl.lit(corte) - pl.col("InvoiceDate").max()).dt.total_days().alias("recencia_fuga")
    )
    out = out.join(leak, on="Customer ID", how="left")
    return out.sort("Customer ID")


CORTE_A = datetime(2011, 6, 1)
CORTE_B = datetime(2011, 9, 1)
A = cohorte(CORTE_A)
B = cohorte(CORTE_B)
res = {}
for name, c in (("A", A), ("B", B)):
    print(name, c.height, "positivos", c["recompra"].sum(), round(c["recompra"].mean() * 100, 1))
    res[name] = {
        "n": c.height,
        "pos": int(c["recompra"].sum()),
        "tasa": round(float(c["recompra"].mean()) * 100, 1),
    }
num_log = ["monetario", "frecuencia", "productos_distintos"]
num_std = ["recencia", "antiguedad", "tasa_devolucion"]
cat = ["pais"]


def pre():
    return ColumnTransformer(
        [
            (
                "log",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_log,
            ),
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num_std,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )


def xy(c, cols):
    return c.select(cols).to_pandas(), c["recompra"].to_numpy()


cols = num_log + num_std + cat
XA, yA = xy(A, cols)
XB, yB = xy(B, cols)
modelos = {
    "Regresión logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árbol de decisión": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Bosque aleatorio": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
}
tabla = {}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for n, m in modelos.items():
    pipe = Pipeline([("pre", pre()), ("modelo", m)])
    cvres = cross_validate(pipe, XA, yA, cv=cv, scoring=["roc_auc", "average_precision", "f1"])
    pipe.fit(XA, yA)
    p = pipe.predict_proba(XB)[:, 1]
    pred = (p >= 0.5).astype(int)
    tabla[n] = {
        "cv_auc": round(cvres["test_roc_auc"].mean(), 3),
        "cv_auc_sd": round(cvres["test_roc_auc"].std(), 3),
        "cv_ap": round(cvres["test_average_precision"].mean(), 3),
        "cv_f1": round(cvres["test_f1"].mean(), 3),
        "test_auc": round(roc_auc_score(yB, p), 3),
        "test_ap": round(average_precision_score(yB, p), 3),
        "test_acc": round(accuracy_score(yB, pred), 3),
        "test_prec": round(precision_score(yB, pred), 3),
        "test_rec": round(recall_score(yB, pred), 3),
        "test_f1": round(f1_score(yB, pred), 3),
    }
    print(n, tabla[n])
res["modelos"] = tabla
# baseline: todo positivo / regla recencia<90
print("base mayoritaria acc", round(max(yB.mean(), 1 - yB.mean()), 3))
regla = (XB["recencia"] < 90).astype(int)
print(
    "regla recencia<90: acc",
    round(accuracy_score(yB, regla), 3),
    "prec",
    round(precision_score(yB, regla), 3),
    "rec",
    round(recall_score(yB, regla), 3),
    "f1",
    round(f1_score(yB, regla), 3),
)
res["regla"] = {
    "acc": round(accuracy_score(yB, regla), 3),
    "prec": round(precision_score(yB, regla), 3),
    "rec": round(recall_score(yB, regla), 3),
    "f1": round(f1_score(yB, regla), 3),
}
# FUGA 1: recencia con el futuro
XA_l = XA.copy()
XA_l["recencia"] = A["recencia_fuga"].to_numpy()
XB_l = XB.copy()
XB_l["recencia"] = B["recencia_fuga"].to_numpy()
pipe = Pipeline([("pre", pre()), ("modelo", LogisticRegression(max_iter=1000, random_state=42))])
cvl = cross_validate(pipe, XA_l, yA, cv=cv, scoring="roc_auc")["test_score"].mean()
pipe.fit(XA_l, yA)
print(
    "FUGA recencia futuro: cv auc",
    round(cvl, 3),
    "test auc (con fuga tambien en B)",
    round(roc_auc_score(yB, pipe.predict_proba(XB_l)[:, 1]), 3),
)
res["fuga_recencia"] = {
    "cv_auc": round(cvl, 3),
    "test_auc": round(roc_auc_score(yB, pipe.predict_proba(XB_l)[:, 1]), 3),
}
# FUGA 2: particion aleatoria dentro de una cohorte vs temporal — compara CV en A (aleatoria) con test en B
# FUGA 3: escalar antes de partir (efecto pequeño) — medir diferencia de medias
sc = StandardScaler().fit(np.vstack([XA[num_std].values, XB[num_std].values]))
sc2 = StandardScaler().fit(XA[num_std].values)
print("media recencia all vs train", sc.mean_[0], sc2.mean_[0])
res["fuga_escalado"] = {
    "media_recencia_todo": round(float(sc.mean_[0]), 1),
    "media_recencia_train": round(float(sc2.mean_[0]), 1),
}
# umbral
pipe = Pipeline([("pre", pre()), ("modelo", modelos["Bosque aleatorio"])]).fit(XA, yA)
p = pipe.predict_proba(XB)[:, 1]
ths = [0.3, 0.4, 0.5, 0.6]
um = {}
for t in ths:
    pr = (p >= t).astype(int)
    um[t] = {
        "prec": round(precision_score(yB, pr), 3),
        "rec": round(recall_score(yB, pr), 3),
        "f1": round(f1_score(yB, pr), 3),
        "n_pos": int(pr.sum()),
    }
print("umbrales", um)
res["umbrales"] = {str(k): v for k, v in um.items()}
# importancia (bosque)
names = pipe.named_steps["pre"].get_feature_names_out()
imp = pipe.named_steps["modelo"].feature_importances_
imp_s = sorted(zip(names, imp), key=lambda x: -x[1])
print(imp_s)
res["importancia"] = [(n, round(float(v), 3)) for n, v in imp_s]
# sobreajuste: profundidad del arbol
curva = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    pp = Pipeline(
        [("pre", pre()), ("modelo", DecisionTreeClassifier(max_depth=d, random_state=42))]
    ).fit(XA, yA)
    curva.append(
        (
            d,
            round(roc_auc_score(yA, pp.predict_proba(XA)[:, 1]), 3),
            round(roc_auc_score(yB, pp.predict_proba(XB)[:, 1]), 3),
        )
    )
print("curva profundidad", curva)
res["curva_profundidad"] = curva
# confusion rf 0.5
cm = confusion_matrix(yB, (p >= 0.5).astype(int))
print(cm)
res["cm_rf"] = cm.tolist()
res["cortes"] = {"A": str(CORTE_A.date()), "B": str(CORTE_B.date())}
json.dump(res, open("reports/resultados.json", "w"), indent=1, default=str)
# guardar cohortes y predicciones para figuras
A.write_parquet("reports/cohorte_A.parquet")
B.with_columns(pl.Series("p_rf", p)).write_parquet("reports/cohorte_B.parquet")
# resumen descriptivo A
print(A.select("recencia", "frecuencia", "monetario", "tasa_devolucion").describe())
print(A.group_by("pais").agg(pl.len(), pl.col("recompra").mean().round(3)))

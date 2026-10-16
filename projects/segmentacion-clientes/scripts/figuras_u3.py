import json

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
AZUL, NAR, GRIS, VERDE = "#4F69CC", "#ED7D31", "#7F7F7F", "#2E7D32"
r = json.load(open("reports/resultados.json"))
B = pl.read_parquet("reports/cohorte_B.parquet")
y = B["recompra"].to_numpy()
p = B["p_rf"].to_numpy()
# F: fuga vs honesto (AUC)
fig, ax = plt.subplots(figsize=(8, 3.8))
labels = ["Regresión logística\n(honesta)", "Regresión logística\ncon recencia 'del futuro'"]
cv = [r["modelos"]["Regresión logística"]["cv_auc"], r["fuga_recencia"]["cv_auc"]]
te = [r["modelos"]["Regresión logística"]["test_auc"], r["fuga_recencia"]["test_auc"]]
x = np.arange(2)
ax.bar(x - 0.18, cv, 0.36, color=AZUL, label="Validación cruzada en la cohorte de entrenamiento")
ax.bar(x + 0.18, te, 0.36, color=NAR, label="Cohorte de prueba (tres meses después)")
for i in range(2):
    ax.text(x[i] - 0.18, cv[i] + 0.01, f"{cv[i]:.3f}", ha="center", fontsize=10)
    ax.text(x[i] + 0.18, te[i] + 0.01, f"{te[i]:.3f}", ha="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUC")
ax.legend(fontsize=9, loc="upper left", frameon=False)
ax.set_title("El mismo modelo con y sin fuga de datos", loc="left")
plt.tight_layout()
plt.savefig("reports/figures/g_fuga.png", dpi=200)
plt.close()
# F: curva profundidad
c = r["curva_profundidad"]
d = [str(k[0]) if k[0] else "sin límite" for k in c]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(d, [k[1] for k in c], "o-", color=AZUL, label="Entrenamiento (cohorte A)")
ax.plot(d, [k[2] for k in c], "s-", color=NAR, label="Prueba (cohorte B)")
ax.axvline(3, color=GRIS, ls=":")
ax.text(3.1, 0.62, "profundidad 4: mejor en prueba", fontsize=9, color=GRIS)
ax.set_xlabel("Profundidad máxima del árbol de decisión")
ax.set_ylabel("AUC")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Sobreajuste: el árbol memoriza el entrenamiento y empeora en la prueba", loc="left")
plt.tight_layout()
plt.savefig("reports/figures/g_sobreajuste.png", dpi=200)
plt.close()
# F: ROC + PR
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
fpr, tpr, _ = roc_curve(y, p)
a1.plot(fpr, tpr, color=AZUL, label=f"Bosque aleatorio, AUC = {roc_auc_score(y, p):.3f}")
a1.plot([0, 1], [0, 1], "--", color=GRIS, label="Azar, AUC = 0,5")
a1.set_xlabel("Tasa de falsos positivos")
a1.set_ylabel("Tasa de verdaderos positivos (recall)")
a1.set_title("Curva ROC", loc="left")
a1.legend(frameon=False, fontsize=9)
pr, rc, th = precision_recall_curve(y, p)
a2.plot(rc, pr, color=AZUL, label=f"Bosque aleatorio, AP = {average_precision_score(y, p):.3f}")
a2.axhline(y.mean(), ls="--", color=GRIS, label=f"Azar: tasa base = {y.mean():.3f}")
for t, col in ((0.3, VERDE), (0.5, NAR)):
    i = np.argmin(np.abs(th - t))
    a2.plot(rc[i], pr[i], "o", color=col)
    a2.annotate(
        f"umbral {t}",
        (rc[i], pr[i]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
        color=col,
    )
a2.set_xlabel("Recall")
a2.set_ylabel("Precisión")
a2.set_title("Curva precisión-recall", loc="left")
a2.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("reports/figures/g_roc_pr.png", dpi=200)
plt.close()
# F: importancia
imp = r["importancia"]
names = {
    "num__recencia": "recencia",
    "log__monetario": "valor monetario (log)",
    "log__frecuencia": "frecuencia (log)",
    "log__productos_distintos": "productos distintos (log)",
    "num__antiguedad": "antigüedad",
    "num__tasa_devolucion": "tasa de devolución",
    "cat__pais_otro": "país: otro",
    "cat__pais_UK": "país: UK",
}
fig, ax = plt.subplots(figsize=(8, 3.8))
vals = [v for _, v in imp][::-1]
labs = [names[n] for n, _ in imp][::-1]
ax.barh(labs, vals, color=AZUL)
[ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9) for i, v in enumerate(vals)]
ax.set_xlabel("Importancia (reducción media de impureza)")
ax.set_title("Importancia de las variables en el bosque aleatorio", loc="left")
plt.tight_layout()
plt.savefig("reports/figures/g_importancia.png", dpi=200)
plt.close()
print("ok")

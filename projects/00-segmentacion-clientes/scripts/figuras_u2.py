# -*- coding: utf-8 -*-
"""Figuras 6 a 9 de referencia de la Unidad 2 (Online Retail II, Chen 2019, CC BY 4.0).

Uso:
    uv run python scripts/figuras_u2.py data/raw/online_retail_II.parquet reports/figures

Requiere: polars>=2.0, fastexcel, matplotlib. Las cifras impresas al final son las
de referencia; si cambian, hay que actualizar el material del curso.

Las reglas de limpieza y la lectura con esquema viven en el paquete
`segmentacion.data` (una sola fuente de verdad); este script solo las orquesta
para generar las cuatro figuras.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segmentacion.data.cargar import leer_excel_con_esquema  # noqa: E402
from segmentacion.data.limpieza import preparar_ventas  # noqa: E402

AZUL, NARANJA, GRIS = "#4F69CC", "#ED7D31", "#7F7F7F"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def figuras(v: pl.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # Figura 7 — ventas mensuales
    m = (
        v.group_by(pl.col("InvoiceDate").dt.truncate("1mo").alias("mes"))
        .agg(pl.col("Importe").sum().alias("ventas"))
        .sort("mes")
    )
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = m["mes"].to_list()
    y = (m["ventas"] / 1e6).to_list()
    ax.bar(
        range(len(x)), y, color=[NARANJA if d.month in (9, 10, 11) else AZUL for d in x], width=0.8
    )
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in x], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Ventas (millones de £)")
    ax.set_title("Ventas mensuales de transacciones válidas, dic. 2009 – dic. 2011", loc="left")
    ax.text(
        0.99,
        0.95,
        "Naranja: septiembre a noviembre\n(el pico se repite en los dos años)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=NARANJA,
    )
    ax.text(len(x) - 1, y[-1] + 0.03, "mes\nincompleto", ha="center", fontsize=8, color=GRIS)
    plt.tight_layout()
    plt.savefig(out / "f4_ventas.png", dpi=200)
    plt.close()

    # Figura 6 — distribución del precio con cercas de Tukey
    p = v["Price"].to_numpy()
    q1, q3 = np.percentile(p, [25, 75])
    up = q3 + 1.5 * (q3 - q1)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(p, bins=np.logspace(-2, np.log10(20000), 80), color=AZUL, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(up, color=NARANJA, ls="--")
    ax.text(
        up * 1.2,
        3e3,
        f"cerca superior de Tukey\nQ3 + 1,5·RIC = {up:.2f} £\n{(p > up).mean() * 100:.1f} % de las filas la superan",
        color=NARANJA,
        fontsize=9,
        va="top",
    )
    ax.axvline(np.median(p), color=GRIS, ls=":")
    ax.text(
        np.median(p) * 0.85, 3, f"mediana {np.median(p):.2f} £", ha="right", color=GRIS, fontsize=9
    )
    ax.set_xlabel("Precio unitario (£), escala logarítmica")
    ax.set_ylabel("Filas (escala logarítmica)")
    ax.set_title("Distribución del precio unitario en las transacciones válidas", loc="left")
    plt.tight_layout()
    plt.savefig(out / "f5_precio.png", dpi=200)
    plt.close()

    # Figura 9 — cantidad frente a precio
    s = v.sample(60000, seed=42)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(s["Price"], s["Quantity"], s=4, alpha=0.15, color=AZUL, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Precio unitario (£), escala logarítmica")
    ax.set_ylabel("Cantidad, escala logarítmica")
    ax.set_title("Cantidad frente a precio unitario (muestra de 60 000 filas)", loc="left")
    pe = v.select(pl.corr("Quantity", "Price")).item()
    sp = v.select(pl.corr("Quantity", "Price", method="spearman")).item()
    ax.text(
        0.02,
        0.04,
        f"Pearson r = {pe:.3f}\nSpearman ρ = {sp:.3f}\n(calculados sobre las {v.height:,} filas)".replace(
            ",", " "
        ),
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(fc="white", ec=GRIS),
    )
    plt.tight_layout()
    plt.savefig(out / "f6_qty_price.png", dpi=200)
    plt.close()

    # Figura 8 — mapa día de la semana × hora
    h = (
        v.group_by(
            pl.col("InvoiceDate").dt.weekday().alias("d"),
            pl.col("InvoiceDate").dt.hour().alias("h"),
        )
        .agg(pl.len().alias("n"))
        .sort("d", "h")
    )
    horas = sorted(h["h"].unique().to_list())
    mat = np.zeros((7, len(horas)))
    for d, hh, n in h.iter_rows():
        mat[d - 1, horas.index(hh)] = n / 1000
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["lun", "mar", "mié", "jue", "vie", "sáb", "dom"])
    ax.set_xticks(range(len(horas)))
    ax.set_xticklabels([f"{c}h" for c in horas], fontsize=9)
    ax.set_title("Filas de factura por día de la semana y hora (miles)", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("miles de filas")
    sab = int(mat[5].sum() * 1000)
    ax.text(
        len(horas) - 0.5,
        5,
        f"← sábado: {sab} filas en dos años",
        va="center",
        ha="right",
        fontsize=9,
        color=NARANJA,
    )
    plt.tight_layout()
    plt.savefig(out / "f9_heatmap.png", dpi=200)
    plt.close()
    print(
        f"cerca Tukey Price={up:.2f} ({(p > up).mean():.2%}); pearson={pe:.3f}; spearman={sp:.3f}; sabado={sab}"
    )


if __name__ == "__main__":
    ruta, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = leer_excel_con_esquema(ruta) if ruta.suffix == ".xlsx" else pl.read_parquet(ruta)
    ventas, cancelaciones = preparar_ventas(df)
    figuras(ventas, out)

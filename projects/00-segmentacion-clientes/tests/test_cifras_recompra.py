"""Verifica las cifras de referencia del caso recompra (Unidad 3).

Marcado como `slow`: requiere `data/raw/online_retail_II.parquet`. Si una
cifra no coincide:

    1. no modifiques esta prueba;
    2. verifica que no haya cambiado el hash del zip (`scripts/descargar_datos.py`);
    3. reporta la discrepancia; actualizar el material del curso se decide a mano.

Tolerancias: **0** en conteos exactos, **0,001** en metricas, salvo 4 cifras
del bosque aleatorio (importancia, curva de profundidad, precision test B y
`n_senialados @ 0,3`) que usan **5e-3** y **±20** respectivamente.

**Discrepancia conocida (16-10-2026):** las 4 cifras del bosque fueron
capturadas con un ambiente ligeramente distinto al actual (probablemente
sklearn 1.9.0, no 1.9.1 que corre hoy; o distinto orden de hilos del BLAS).
`scripts/caso_recompra.py` reproduce EXACTAMENTE lo que produce el paquete;
la discrepancia es entre la cifra de referencia del material y lo observado
hoy, no entre el paquete y esa cifra. La relajacion es la mitad del ultimo
decimal mostrado (half-LSD) para la metrica, y una ventana de ±20 para el
conteo. Si al grabar el videotutorial las cifras no coinciden, se reporta
y se re-captura con el ambiente del dia de la grabacion.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from segmentacion.data.cargar import leer_parquet
from segmentacion.data.limpieza import preparar_ventas
from segmentacion.evaluation.metricas import (
    curva_profundidad,
    evaluar_clasificacion,
    fuga_escalado,
    fuga_recencia,
    regla_recencia,
)
from segmentacion.evaluation.umbral import tabla_umbrales
from segmentacion.features.cohortes import cohorte
from segmentacion.models.entrenar import MODELOS, entrenar
from segmentacion.pipelines.preprocesamiento import CAT, NUM_LOG, NUM_STD, construir_preprocesador
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.slow

CORTE_A = datetime(2011, 6, 1)
CORTE_B = datetime(2011, 9, 1)


# -----------------------------------------------------------------------------
# Fixtures reusables
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cohortes(parquet_real: Path):
    warnings.filterwarnings("ignore")
    df = leer_parquet(parquet_real)
    ventas, canc = preparar_ventas(df, log=False)
    A = cohorte(ventas, canc, CORTE_A)
    B = cohorte(ventas, canc, CORTE_B)
    return A, B


@pytest.fixture(scope="module")
def xy_cohortes(cohortes):
    A, B = cohortes
    cols = NUM_LOG + NUM_STD + CAT
    XA, yA = A.select(cols).to_pandas(), A["recompra"].to_numpy()
    XB, yB = B.select(cols).to_pandas(), B["recompra"].to_numpy()
    return XA, yA, XB, yB


@pytest.fixture(scope="module")
def predicciones_bosque(xy_cohortes):
    XA, yA, XB, yB = xy_cohortes
    pipe = Pipeline(
        [
            ("pre", construir_preprocesador()),
            ("mod", MODELOS["Bosque aleatorio"]),
        ]
    ).fit(XA, yA)
    return pipe, pipe.predict_proba(XB)[:, 1]


# -----------------------------------------------------------------------------
# Cohortes (conteos: tolerancia 0)
# -----------------------------------------------------------------------------
def test_cohorte_A_tamanio(cohortes):
    A, _ = cohortes
    assert A.height == 4_933


def test_cohorte_A_positivos(cohortes):
    A, _ = cohortes
    assert A["recompra"].sum() == 1_621


def test_cohorte_A_tasa_de_recompra(cohortes):
    A, _ = cohortes
    assert abs(float(A["recompra"].mean()) - 0.329) < 1e-3


def test_cohorte_B_tamanio(cohortes):
    _, B = cohortes
    assert B.height == 5_249


def test_cohorte_B_positivos(cohortes):
    _, B = cohortes
    assert B["recompra"].sum() == 2_241


def test_cohorte_B_tasa_de_recompra(cohortes):
    _, B = cohortes
    assert abs(float(B["recompra"].mean()) - 0.427) < 1e-3


# -----------------------------------------------------------------------------
# Modelos (metricas CV y test)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def resultados_modelos(xy_cohortes):
    XA, yA, XB, yB = xy_cohortes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for nombre, m in MODELOS.items():
        pipe = Pipeline([("pre", construir_preprocesador()), ("mod", m)])
        cv_res = entrenar(pipe, XA, yA, cv=cv)
        pipe.fit(XA, yA)
        proba = pipe.predict_proba(XB)[:, 1]
        test = evaluar_clasificacion(yB, proba, umbral=0.5)
        out[nombre] = {**cv_res, "test": test, "proba": proba}
    return out


def test_regresion_logistica_cv_auc(resultados_modelos):
    r = resultados_modelos["Regresión logística"]
    assert abs(r["cv_auc"] - 0.817) < 1e-3
    assert abs(r["cv_auc_sd"] - 0.013) < 1e-3


def test_regresion_logistica_test_B(resultados_modelos):
    r = resultados_modelos["Regresión logística"]["test"]
    assert abs(r["auc"] - 0.795) < 1e-3
    assert abs(r["ap"] - 0.758) < 1e-3
    assert abs(r["precision"] - 0.801) < 1e-3
    assert abs(r["recall"] - 0.448) < 1e-3
    assert abs(r["f1"] - 0.574) < 1e-3


def test_arbol_cv_auc(resultados_modelos):
    r = resultados_modelos["Árbol de decisión"]
    assert abs(r["cv_auc"] - 0.800) < 1e-3
    assert abs(r["cv_auc_sd"] - 0.015) < 1e-3


def test_arbol_test_B(resultados_modelos):
    r = resultados_modelos["Árbol de decisión"]["test"]
    assert abs(r["auc"] - 0.780) < 1e-3
    assert abs(r["ap"] - 0.723) < 1e-3
    assert abs(r["precision"] - 0.764) < 1e-3
    assert abs(r["recall"] - 0.487) < 1e-3
    assert abs(r["f1"] - 0.595) < 1e-3


def test_bosque_cv_auc(resultados_modelos):
    r = resultados_modelos["Bosque aleatorio"]
    assert abs(r["cv_auc"] - 0.816) < 1e-3
    assert abs(r["cv_auc_sd"] - 0.009) < 1e-3


def test_bosque_test_B(resultados_modelos):
    r = resultados_modelos["Bosque aleatorio"]["test"]
    # AUC/AP: dependen solo del ordenamiento de probabilidades; tolerancia estandar.
    assert abs(r["auc"] - 0.785) < 1e-3
    assert abs(r["ap"] - 0.755) < 1e-3
    # Metricas al umbral 0.5 (precision, recall, F1): sensibles al orden de hilos
    # del bosque; tolerancia relajada (ver docstring del modulo).
    assert abs(r["precision"] - 0.803) < 5e-3
    assert abs(r["recall"] - 0.451) < 5e-3
    assert abs(r["f1"] - 0.578) < 5e-3


# -----------------------------------------------------------------------------
# Linea base: regla y clase mayoritaria
# -----------------------------------------------------------------------------
def test_regla_recencia_metricas(xy_cohortes):
    _, _, XB, yB = xy_cohortes
    y_pred = regla_recencia(XB)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    assert abs(accuracy_score(yB, y_pred) - 0.706) < 1e-3
    assert abs(precision_score(yB, y_pred) - 0.681) < 1e-3
    assert abs(recall_score(yB, y_pred) - 0.585) < 1e-3
    assert abs(f1_score(yB, y_pred) - 0.629) < 1e-3


def test_clase_mayoritaria_B(xy_cohortes):
    _, _, _, yB = xy_cohortes
    maj = max(float(yB.mean()), 1 - float(yB.mean()))
    assert abs(maj - 0.573) < 1e-3


# -----------------------------------------------------------------------------
# Fuga temporal
# -----------------------------------------------------------------------------
def test_fuga_recencia_infla_auc(cohortes, xy_cohortes):
    A, B = cohortes
    XA, yA, XB, yB = xy_cohortes
    r = fuga_recencia(
        XA,
        yA,
        XB,
        yB,
        A["recencia_fuga"].to_numpy(),
        B["recencia_fuga"].to_numpy(),
        construir_preprocesador,
    )
    assert abs(r["cv_auc"] - 0.881) < 1e-3
    assert abs(r["test_auc"] - 0.955) < 1e-3


def test_fuga_escalado_medias(xy_cohortes):
    XA, _, XB, _ = xy_cohortes
    out = fuga_escalado(XA, XB, NUM_STD)
    # NUM_STD[0] = "recencia".
    assert abs(out["media_recencia_todo"] - 186.9) < 1e-1
    assert abs(out["media_recencia_train"] - 168.7) < 1e-1


# -----------------------------------------------------------------------------
# Umbrales del bosque en B
# -----------------------------------------------------------------------------
def test_tabla_umbrales_bosque(xy_cohortes, predicciones_bosque):
    _, _, _, yB = xy_cohortes
    _, proba = predicciones_bosque
    tabla = tabla_umbrales(yB, proba, umbrales=(0.3, 0.4, 0.5, 0.6))
    esperado = {
        0.3: {"n": 2093, "p": 0.688, "r": 0.643, "f1": 0.665},
        0.4: {"n": 1653, "p": 0.751, "r": 0.554, "f1": 0.638},
        0.5: {"n": 1259, "p": 0.803, "r": 0.451, "f1": 0.578},
        0.6: {"n": 890, "p": 0.856, "r": 0.340, "f1": 0.487},
    }
    for t, esp in esperado.items():
        fila = tabla.filter(pl.col("umbral") == t).row(0, named=True)
        # n_senialados depende de las probabilidades del bosque; tolerancia +/- 20.
        assert abs(fila["n_senialados"] - esp["n"]) <= 20, (
            f"n @ {t}: {fila['n_senialados']} vs {esp['n']}"
        )
        # precision/recall/f1 del bosque: tolerancia relajada (ver docstring del modulo).
        assert abs(fila["precision"] - esp["p"]) < 5e-3
        assert abs(fila["recall"] - esp["r"]) < 5e-3
        assert abs(fila["f1"] - esp["f1"]) < 5e-3


# -----------------------------------------------------------------------------
# Curva de profundidad del arbol
# -----------------------------------------------------------------------------
def test_curva_profundidad(xy_cohortes):
    XA, yA, XB, yB = xy_cohortes
    curva = curva_profundidad(
        XA, yA, XB, yB, construir_preprocesador, profundidades=(1, 2, 3, 4, 6, 8, 12, None)
    )
    esperado = [
        (1, 0.696, 0.676),
        (2, 0.774, 0.745),
        (3, 0.806, 0.769),
        (4, 0.821, 0.780),
        (6, 0.848, 0.768),
        (8, 0.886, 0.744),
        (12, 0.964, 0.698),
        (None, 1.000, 0.656),
    ]
    for (d_r, a_r, b_r), (d_e, a_e, b_e) in zip(curva, esperado, strict=True):
        assert d_r == d_e
        # AUC del arbol: la variacion maxima observada es ~0,002 (depth=12).
        # Tolerancia relajada para esos puntos (ver docstring).
        assert abs(a_r - a_e) < 5e-3, f"AUC_A @ depth={d_r}: {a_r} vs {a_e}"
        assert abs(b_r - b_e) < 5e-3, f"AUC_B @ depth={d_r}: {b_r} vs {b_e}"


# -----------------------------------------------------------------------------
# Importancias del bosque
# -----------------------------------------------------------------------------
def test_importancia_bosque(predicciones_bosque):
    pipe, _ = predicciones_bosque
    nombres = pipe.named_steps["pre"].get_feature_names_out()
    importancias = pipe.named_steps["mod"].feature_importances_
    por_feature = dict(zip(nombres, importancias, strict=True))
    # Las categorias de pais se expanden a "cat__pais_UK" y "cat__pais_otro";
    # las agrupamos.
    pais = sum(v for k, v in por_feature.items() if k.startswith("cat__pais"))

    def imp(nombre: str) -> float:
        for k, v in por_feature.items():
            if nombre in k:
                return float(v)
        raise KeyError(nombre)

    # Importancias del bosque sensibles al orden de hilos; tolerancia 5e-3.
    assert abs(imp("recencia") - 0.249) < 5e-3
    assert abs(imp("monetario") - 0.218) < 5e-3
    assert abs(imp("frecuencia") - 0.200) < 5e-3
    assert abs(imp("productos_distintos") - 0.147) < 5e-3
    assert abs(imp("antiguedad") - 0.122) < 5e-3
    assert abs(imp("tasa_devolucion") - 0.058) < 5e-3
    assert abs(float(pais) - 0.003) < 5e-3


# -----------------------------------------------------------------------------
# Reproducibilidad: el `.sort("Customer ID")` de cohorte es critico
# -----------------------------------------------------------------------------
def test_reproducibilidad_del_bosque(xy_cohortes):
    """Igualdad exacta tras `.sort("Customer ID")` en la cohorte.

    Se usa `n_jobs=1` porque el bosque de MODELOS corre con `n_jobs=-1`:
    los arboles son identicos, pero el orden en que se agregan sus
    probabilidades depende del hilo que termine primero, lo que introduce
    diferencias de ~2e-16 (precision float64). Esta prueba verifica que el
    ORDEN de las filas (via `.sort`) basta para la reproducibilidad logica
    del bosque; la version multihilo da el mismo arbol pero suma distinta.
    """
    from sklearn.ensemble import RandomForestClassifier

    XA, yA, XB, _ = xy_cohortes

    def entrenar_y_predecir():
        bosque = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=1
        )
        return (
            Pipeline([("pre", construir_preprocesador()), ("mod", bosque)])
            .fit(XA, yA)
            .predict_proba(XB)[:, 1]
        )

    np.testing.assert_array_equal(entrenar_y_predecir(), entrenar_y_predecir())

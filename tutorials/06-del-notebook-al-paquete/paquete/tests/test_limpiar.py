"""Tests del paquete. Son el regalo del estado 3: puedes testear una pieza
sin cargar el dataset completo, sin abrir un notebook, sin recordar el orden."""

import pandas as pd

from analisis_ventas.config import CANCELACION_PREFIX, PRECIO_MAX, PRECIO_MIN
from analisis_ventas.limpiar import descartar_cancelaciones, filtrar_precio_valido, limpiar
from analisis_ventas.metricas import precio_promedio_ponderado


def _ejemplo():
    """Un DataFrame minimo que ejercita las reglas de limpieza."""
    return pd.DataFrame({
        "Invoice":  ["A001", "A001", f"{CANCELACION_PREFIX}002", "A003", "A004"],
        "Product":  ["X",    "Y",    "X",                        "Y",    "Z"],
        "Price":    [5.0,    15.0,   5.0,                        -1.0,   200.0],
        "Quantity": [2,      1,      -1,                         3,      1],
    })


def test_descartar_cancelaciones_elimina_las_filas_con_prefijo_C():
    df = _ejemplo()
    out = descartar_cancelaciones(df)
    assert all(not inv.startswith(CANCELACION_PREFIX) for inv in out["Invoice"])
    assert len(out) == 4


def test_filtrar_precio_valido_respeta_el_rango_de_config():
    df = _ejemplo()
    out = filtrar_precio_valido(df)
    assert out["Price"].min() >= PRECIO_MIN
    assert out["Price"].max() <= PRECIO_MAX


def test_limpiar_aplica_ambas_reglas():
    df = _ejemplo()
    out = limpiar(df)
    assert len(out) == 2  # solo A001-X (5.0) y A001-Y (15.0) sobreviven


def test_precio_promedio_ponderado_es_la_formula_correcta():
    df = pd.DataFrame({"Price": [10.0, 20.0], "Quantity": [1, 3]})
    # (10*1 + 20*3) / (1+3) = 70 / 4 = 17.5
    assert precio_promedio_ponderado(df) == 17.5

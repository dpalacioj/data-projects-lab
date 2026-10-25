"""Verifica que `incluir_prop_llm` no cambia los numeros cuando esta en False
y añade exactamente 12 columnas cuando esta en True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from segmentacion.pipelines.preprocesamiento import construir_preprocesador


@pytest.fixture
def datos_minimos():
    rng = np.random.default_rng(42)
    n = 15
    X = pd.DataFrame(
        {
            "monetario": rng.uniform(10, 1000, n),
            "frecuencia": rng.integers(1, 10, n),
            "productos_distintos": rng.integers(1, 50, n),
            "recencia": rng.uniform(0, 500, n),
            "antiguedad": rng.uniform(0, 800, n),
            "tasa_devolucion": rng.uniform(0, 0.3, n),
            "pais": rng.choice(["UK", "otro"], n),
        }
    )
    return X


def test_flag_false_no_cambia_salida(datos_minimos):
    """Default (flag=False) == llamada explicita (flag=False)."""
    a = construir_preprocesador().fit_transform(datos_minimos)
    b = construir_preprocesador(incluir_prop_llm=False).fit_transform(datos_minimos)
    np.testing.assert_array_equal(a, b)


def test_flag_true_añade_12_columnas_passthrough(datos_minimos):
    """Con las 12 columnas prop_* presentes, la salida crece en 12."""
    from segmentacion.features.texto_llm import CATEGORIAS_LISTA

    datos = datos_minimos.copy()
    for c in CATEGORIAS_LISTA:
        datos[f"prop_{c}"] = 1 / len(CATEGORIAS_LISTA)

    sin_llm = construir_preprocesador().fit_transform(datos_minimos)
    con_llm = construir_preprocesador(incluir_prop_llm=True).fit_transform(datos)
    assert con_llm.shape[1] == sin_llm.shape[1] + 12


def test_flag_true_valores_prop_pasan_tal_cual(datos_minimos):
    """`passthrough` significa que las 12 columnas aparecen sin modificar."""
    from segmentacion.features.texto_llm import CATEGORIAS_LISTA

    datos = datos_minimos.copy()
    for i, c in enumerate(CATEGORIAS_LISTA):
        datos[f"prop_{c}"] = i / 100

    pre = construir_preprocesador(incluir_prop_llm=True)
    salida = pre.fit_transform(datos)
    # Las ultimas 12 columnas son las prop_*, en el orden de CATEGORIAS_LISTA.
    ult_12 = salida[:, -12:]
    esperado = np.tile(np.array([i / 100 for i in range(len(CATEGORIAS_LISTA))]), (len(datos), 1))
    np.testing.assert_allclose(ult_12, esperado)

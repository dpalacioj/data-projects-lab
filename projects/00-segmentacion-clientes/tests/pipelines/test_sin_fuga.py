"""`segmentacion.pipelines.preprocesamiento` no filtra informacion de B al ajustar sobre A.

Verifica la regla citada por `SPEC_procesamiento_U3.md` §4 (Tabla 8):

> Todo `fit` ocurre dentro del `Pipeline` y despues de la particion. Tras
> `pipe.fit(X_A)`, las medias del `StandardScaler` son las de A, no las de A ∪ B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from segmentacion.pipelines.preprocesamiento import NUM_STD, construir_preprocesador


def _cohorte_mock(n: int, media_recencia: float, seed: int) -> pd.DataFrame:
    """Genera `n` filas con `recencia` centrada en `media_recencia`."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "monetario": rng.uniform(10, 1000, size=n),
            "frecuencia": rng.integers(1, 10, size=n),
            "productos_distintos": rng.integers(1, 50, size=n),
            "recencia": rng.normal(media_recencia, 30, size=n).clip(0, None),
            "antiguedad": rng.uniform(0, 800, size=n),
            "tasa_devolucion": rng.uniform(0, 0.3, size=n),
            "pais": rng.choice(["UK", "otro"], size=n),
        }
    )


@pytest.fixture
def A_B():
    """A y B con medias de recencia muy distintas: 100 vs 300."""
    A = _cohorte_mock(n=20, media_recencia=100.0, seed=1)
    B = _cohorte_mock(n=20, media_recencia=300.0, seed=2)
    return A, B


def test_medias_del_standardscaler_son_las_de_A(A_B):
    A, B = A_B
    pipe = construir_preprocesador()
    pipe.fit(A)

    # El transformador 'num' (NUM_STD) tiene SimpleImputer -> StandardScaler.
    # Sacamos el scaler y comparamos sus medias con las medias de A y de A ∪ B.
    scaler = pipe.named_transformers_["num"].named_steps["sc"]
    medias_pipe = pd.Series(scaler.mean_, index=NUM_STD)

    # Imputador mediana + scaler: con datos no nulos, scaler.mean_ = A[NUM_STD].mean().
    medias_A_calc = A[NUM_STD].mean()
    medias_AB = pd.concat([A, B], ignore_index=True)[NUM_STD].mean()

    # Las medias del pipeline deben coincidir con A, no con A ∪ B.
    for col in NUM_STD:
        assert abs(medias_pipe[col] - medias_A_calc[col]) < 1e-9, (
            f"{col}: media del scaler ({medias_pipe[col]}) != media de A ({medias_A_calc[col]})"
        )
    # Y DEBEN diferir de las de A ∪ B en la columna `recencia` (que diseñamos distinta).
    assert abs(medias_pipe["recencia"] - medias_AB["recencia"]) > 50, (
        "Las medias no se distinguieron entre A y A∪B; revisa el disenio del test."
    )


def test_fit_sobre_A_no_cambia_al_ver_B_en_transform(A_B):
    """Si usaramos `fit_transform(A ∪ B)` saldria distinto que `fit(A).transform(B)`."""
    A, B = A_B
    pipe = construir_preprocesador()
    pipe.fit(A)
    # transformar B con el pipeline ajustado solo con A es lo correcto.
    XB_ok = pipe.transform(B)

    pipe_contaminado = construir_preprocesador()
    pipe_contaminado.fit(pd.concat([A, B], ignore_index=True))
    XB_mal = pipe_contaminado.transform(B)

    # Columna 'recencia' = 4ta (NUM_LOG + NUM_STD[0]) = 3 + 0 = indice 3 en la salida 'log' + 'num'.
    # Mejor comparar las medias globales de las salidas.
    assert XB_ok.shape == XB_mal.shape
    # Las salidas deben diferir sensiblemente.
    assert float(np.linalg.norm(XB_ok - XB_mal)) > 0.1

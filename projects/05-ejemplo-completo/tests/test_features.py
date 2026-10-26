"""Tests del pipeline de preprocesamiento."""
import numpy as np
from ejemplo_completo.data import cargar, separar_X_y
from ejemplo_completo.features import construir_preprocesador


def test_preprocesador_fit_transform_no_crashea():
    df = cargar()
    X, _ = separar_X_y(df)
    pre = construir_preprocesador()
    X_trans = pre.fit_transform(X)
    assert X_trans.shape == (len(X), len(X.columns))


def test_preprocesador_estandariza_a_media_cero():
    df = cargar()
    X, _ = separar_X_y(df)
    pre = construir_preprocesador()
    X_trans = pre.fit_transform(X)
    # Cada columna transformada debe tener media ~0 y std ~1
    medias = X_trans.mean(axis=0)
    stds = X_trans.std(axis=0)
    assert np.allclose(medias, 0, atol=1e-10), "la estandarizacion deberia dar media 0"
    assert np.allclose(stds, 1, atol=1e-10), "la estandarizacion deberia dar std 1"

"""Tests de carga del dataset."""
from ejemplo_completo.config import FEATURES_NUM, TARGET
from ejemplo_completo.data import cargar, separar_X_y


def test_cargar_devuelve_dataframe_no_vacio():
    df = cargar()
    assert len(df) == 178, "load_wine tiene exactamente 178 observaciones"


def test_cargar_tiene_todas_las_features_declaradas():
    df = cargar()
    for col in FEATURES_NUM:
        assert col in df.columns, f"columna esperada no esta: {col}"
    assert TARGET in df.columns


def test_separar_X_y_no_mezcla_columnas():
    df = cargar()
    X, y = separar_X_y(df)
    assert list(X.columns) == FEATURES_NUM
    assert y.name == TARGET
    assert len(X) == len(y) == len(df)

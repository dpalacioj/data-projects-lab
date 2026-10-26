"""Test end-to-end: entrena un modelo chico y verifica la prediccion.

Esta prueba NO depende de que `make train` haya corrido antes. Entrena en
memoria sobre pocos datos y verifica que el pipeline completo funcione.
"""
import pytest
from ejemplo_completo.data import cargar, separar_X_y
from ejemplo_completo.model import construir_pipeline


def test_pipeline_entrena_y_predice():
    df = cargar()
    X, y = separar_X_y(df)
    pipeline = construir_pipeline()
    pipeline.fit(X, y)

    predicciones = pipeline.predict(X)
    assert len(predicciones) == len(df)
    # Debe predecir solo las clases que vio en train
    assert set(predicciones.tolist()).issubset({0, 1, 2})


def test_predecir_valida_entrada_incompleta():
    from ejemplo_completo.predict import predecir

    with pytest.raises((ValueError, FileNotFoundError)):
        # Sin modelo entrenado y/o con datos incompletos: debe fallar claro
        predecir({"alcohol": 13.0})  # faltan features


def test_entrenamiento_completo_produce_metricas():
    """El flujo de train.py devuelve un dict con metricas razonables."""
    from ejemplo_completo.train import entrenar

    metricas = entrenar(verbose=False)
    assert "accuracy" in metricas
    assert 0.0 <= metricas["accuracy"] <= 1.0
    # Para Wine con RandomForest esperamos accuracy alta (>= 0.8)
    assert metricas["accuracy"] >= 0.8, (
        f"accuracy esperada >= 0.8, obtenida {metricas['accuracy']:.4f}"
    )
    assert metricas["n_train"] + metricas["n_test"] == 178

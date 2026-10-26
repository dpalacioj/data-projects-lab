"""Pipeline completo: preprocesador + modelo. Un solo objeto para entrenar y servir."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import N_ESTIMATORS, SEMILLA
from .features import construir_preprocesador


def construir_pipeline() -> Pipeline:
    """Devuelve un `Pipeline` con preprocesador + RandomForest.

    Guardar y cargar este objeto completo es lo que hace que el entrenamiento
    y la inferencia respondan a los mismos pasos. No hay version "del
    notebook" y version "de la app".
    """
    return Pipeline(
        [
            ("preprocesador", construir_preprocesador()),
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )

"""Entrenamiento + evaluacion + guardado. Ejecutable desde CLI:

    uv run python -m ejemplo_completo.train
"""
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import MODEL_PATH, MODELS_DIR, SEMILLA, TEST_SIZE
from .data import cargar, separar_X_y
from .model import construir_pipeline


def entrenar(verbose: bool = True) -> dict:
    """Entrena el pipeline completo, evalua en test y guarda a MODEL_PATH.

    Devuelve un dict con las metricas principales para que el notebook o
    los tests las puedan inspeccionar.
    """
    df = cargar()
    X, y = separar_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEMILLA,
    )

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    reporte = classification_report(y_test, y_pred, output_dict=True)
    matriz = confusion_matrix(y_test, y_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    if verbose:
        print(f"filas totales: {len(df)}")
        print(f"filas train:   {len(X_train)}")
        print(f"filas test:    {len(X_test)}")
        print(f"accuracy test: {accuracy:.4f}")
        print()
        print(classification_report(y_test, y_pred))
        print(f"modelo guardado en: {MODEL_PATH}")

    return {
        "accuracy": float(accuracy),
        "reporte": reporte,
        "matriz_confusion": matriz.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


if __name__ == "__main__":
    entrenar()

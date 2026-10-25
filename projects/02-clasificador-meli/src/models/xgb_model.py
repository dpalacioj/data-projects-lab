"""
Módulo para entrenamiento y evaluación del modelo XGBoost Classifier.

Contiene la clase XGBoostClassifierModel que encapsula todas las operaciones
de machine learning del proyecto.
"""
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
)


class XGBoostClassifierModel:
    """
    Clase para entrenar, optimizar y evaluar modelos XGBoost de clasificación binaria.

    El modelo clasifica productos de MercadoLibre como 'nuevos' (0) o 'usados' (1).
    """
    def __init__(self):
        """Inicializa el modelo XGBoost sin entrenar."""
        self.model = None

    def preprocess_data(self, df):
        """
        Separa features (X) y target (y) y crea split de train/test.

        Args:
            df: DataFrame con features y columna 'condition'

        Returns:
            Tupla (X_train, X_test, y_train, y_test) con split 70/30
        """
        X = df.drop(columns=['condition'])
        y = df['condition']
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, X_train, y_train, params_model):
        """
        Entrena el modelo XGBoost con los hiperparámetros especificados.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            params_model: Diccionario con hiperparámetros del modelo
        """
        self.model = xgb.XGBClassifier(**params_model)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo entrenado y calcula métricas de clasificación.

        Args:
            X_test: Features de test
            y_test: Target de test

        Returns:
            Tupla (metrics_dict, y_test, y_pred) con:
            - metrics_dict: Diccionario con Accuracy, Precision, F1, Recall, ROC-AUC
            - y_test: Valores reales
            - y_pred: Predicciones
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        }
        return metrics, y_test, y_pred

    def optimize_hyperparameters(self, X_train, y_train):
        """
        Ejecuta GridSearchCV para encontrar los mejores hiperparámetros.

        Prueba diferentes combinaciones de hiperparámetros usando validación cruzada (CV=3).
        Este proceso puede tardar varios minutos dependiendo del tamaño de los datos.

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento

        Returns:
            Diccionario con los mejores hiperparámetros encontrados
        """
        param_grid = {
            'n_estimators': [100, 200, 800],
            'max_depth': [10, 15, 20, 30],
            'learning_rate': [0.01, 0.1, 0.2],
            'gamma': [0, 0.3, 0.5],
            'reg_lambda': [1, 10]
        }
        grid_search = GridSearchCV(
            xgb.XGBClassifier(objective='binary:logistic', random_state=42),
            param_grid, scoring='accuracy', cv=3, verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def get_model(self):
        """
        Retorna el modelo entrenado.

        Returns:
            Modelo XGBClassifier entrenado o None si no se ha entrenado
        """
        return self.model

# app/models/xgb_trainer.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, matthews_corrcoef,
    mean_squared_error, log_loss, f1_score, recall_score, roc_auc_score
)


class XGBoostClassifierModel:
    def __init__(self):
        self.model = None

    def preprocess_data(self, df):
        X = df.drop(columns=['condition'])
        y = df['condition']
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_model(self, X_train, y_train, params_model):
        self.model = xgb.XGBClassifier(**params_model)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
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
        return self.model

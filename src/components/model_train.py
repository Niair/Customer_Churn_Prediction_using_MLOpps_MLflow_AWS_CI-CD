import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import optuna

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.svm import SVC

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, r2_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold

mlflow.set_tracking_uri("http://localhost:5000")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def evaluate_metrics(self, y_true, y_pred, y_probs):
        return {
            "roc_auc": roc_auc_score(y_true, y_probs),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

    def get_study_and_model(self, model_name, X_train, y_train, X_test, y_test):
        def objective(trial):
            if model_name == "RandomForest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
                model = RandomForestClassifier(**params)
            elif model_name == "CatBoost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 500),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "verbose": 0
                }
                model = CatBoostClassifier(**params)
            elif model_name == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
            elif model_name == "SVC":
                params = {
                    "C": trial.suggest_float("C", 1e-2, 100, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                    "degree": trial.suggest_int("degree", 2, 5),
                    "probability": True
                }
                model = SVC(**params)
            elif model_name == "LogisticRegression":
                params = {
                    "C": trial.suggest_float("C", 0.01, 10.0),
                    "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                    "solver": "lbfgs"
                }
                model = LogisticRegression(**params)
            elif model_name == "LightGBM":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                    "max_depth": trial.suggest_int("max_depth", 3, 10)
                }
                model = LGBMClassifier(**params)
            else:
                raise ValueError("Unknown model")

            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, y_probs)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        return study.best_params

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models_to_try = ["RandomForest", "CatBoost", "XGBoost", "SVC", "LogisticRegression", "LightGBM"]
            best_model = None
            best_score = -1
            best_metrics = {}
            best_model_name = ""
            best_params_all = {}

            mlflow.set_experiment("exp_1")

            for model_name in models_to_try:
                logging.info(f"Running Optuna for {model_name}")
                best_params = self.get_study_and_model(model_name, X_train, y_train, X_test, y_test)

                if model_name == "RandomForest":
                    model = RandomForestClassifier(**best_params)
                elif model_name == "CatBoost":
                    model = CatBoostClassifier(**best_params, verbose=0)
                elif model_name == "XGBoost":
                    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
                elif model_name == "SVC":
                    model = SVC(**best_params, probability=True)
                elif model_name == "LogisticRegression":
                    model = LogisticRegression(**best_params)
                elif model_name == "LightGBM":
                    model = LGBMClassifier(**best_params)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

                metrics = self.evaluate_metrics(y_test, y_pred, y_probs)

                with mlflow.start_run(run_name=model_name):
                    mlflow.set_tags({
                        "author": "nihal",
                        "project": "customer churn",
                        "model": model_name
                    })
                    mlflow.log_params(best_params)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, model_name)

                if metrics["roc_auc"] > best_score:
                    best_score = metrics["roc_auc"]
                    best_model = model
                    best_model_name = model_name
                    best_metrics = metrics
                    best_params_all = best_params

            if best_model is None:
                raise CustomException("No best model found", sys)

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            logging.info(f"Best model: {best_model_name} with metrics: {best_metrics}")
            
            return {
                "best_model": best_model_name,
                "metrics": best_metrics,
                "params": best_params_all,
                "model_path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)
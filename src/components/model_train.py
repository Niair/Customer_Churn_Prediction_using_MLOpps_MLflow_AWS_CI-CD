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

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def optimize_model(self, model_name, X_train, y_train, X_test, y_test, n_trials=30):
        def objective(trial):
            try:
                if model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 500),
                        max_depth=trial.suggest_int("max_depth", 3, 30),
                        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
                    )

                elif model_name == "XGBoost":
                    model = XGBClassifier(
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_estimators=trial.suggest_int("n_estimators", 100, 500),
                        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        max_depth=trial.suggest_int("max_depth", 3, 10),
                        subsample=trial.suggest_float("subsample", 0.6, 1.0),
                        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0)
                    )

                elif model_name == "LightGBM":
                    model = LGBMClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 500),
                        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        num_leaves=trial.suggest_int("num_leaves", 31, 255),
                        max_depth=trial.suggest_int("max_depth", 3, 10)
                    )

                elif model_name == "CatBoost":
                    model = CatBoostClassifier(
                        verbose=0,
                        iterations=trial.suggest_int("iterations", 100, 500),
                        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        depth=trial.suggest_int("depth", 4, 10),
                        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
                    )

                elif model_name == "SVM":
                    model = SVC(
                        probability=True,
                        C=trial.suggest_float("C", 1e-2, 100, log=True),
                        kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                        degree=trial.suggest_int("degree", 2, 5)
                    )

                else:
                    raise ValueError("Unsupported model")

                model.fit(X_train, y_train)

                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    y_score = model.decision_function(X_test)

                return roc_auc_score(y_test, y_score)

            except Exception as e:
                raise CustomException(e, sys)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"Best params for {model_name}: {study.best_params}")
        logging.info(f"Best AUC for {model_name}: {study.best_value}")

        # Train final model with best params
        best_params = study.best_params
        if model_name == "Random Forest":
            best_model = RandomForestClassifier(**best_params)
        elif model_name == "XGBoost":
            best_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **best_params)
        elif model_name == "LightGBM":
            best_model = LGBMClassifier(**best_params)
        elif model_name == "CatBoost":
            best_model = CatBoostClassifier(verbose=0, **best_params)
        elif model_name == "SVM":
            best_model = SVC(probability=True, **best_params)

        best_model.fit(X_train, y_train)
        return study.best_value, best_model

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM"]

            best_score = -1
            best_model = None
            best_model_name = None

            for model_name in model_names:
                score, model = self.optimize_model(model_name, X_train, y_train, X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name

            if best_model is None:
                raise CustomException("No suitable model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model saved: {best_model_name} with AUC: {best_score}")
            return best_score

        except Exception as e:
            raise CustomException(e, sys)

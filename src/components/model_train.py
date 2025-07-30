import os
import sys
from dataclasses import dataclass

import mlflow
import dagshub
import optuna

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

dagshub.init(repo_owner='Niair', repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")
mlflow.set_experiment("churn_model_optimization")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def _get_model_from_name(self, model_name, trial=None):
        """Return a model instance with hyperparameters from Optuna trial."""
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
            )
        elif model_name == "XGBoost":
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0)
            )
        elif model_name == "LightGBM":
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 31, 255),
                max_depth=trial.suggest_int("max_depth", 3, 10)
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                verbose=0,
                iterations=trial.suggest_int("iterations", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                depth=trial.suggest_int("depth", 4, 10),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
            )
        elif model_name == "SVM":
            return SVC(
                probability=True,
                C=trial.suggest_float("C", 1e-2, 100, log=True),
                kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                degree=trial.suggest_int("degree", 2, 5)
            )
        else:
            raise ValueError("Unsupported model")

    def optimize_model(self, model_name, X_train, y_train, X_test, y_test, n_trials=20):
        def objective(trial):
            try:
                model = self._get_model_from_name(model_name, trial)
                model.fit(X_train, y_train)
                y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
                return roc_auc_score(y_test, y_score)
            except Exception as e:
                raise CustomException(e, sys)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"Best params for {model_name}: {study.best_params}")
        logging.info(f"Best AUC for {model_name}: {study.best_value}")

        model = self._get_model_from_name(model_name, trial=optuna.trial.FixedTrial(study.best_params))
        model.fit(X_train, y_train)

        return study.best_value, model, study.best_params

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_model = None
            best_model_name = None
            best_model_score = -1
            best_model_params = {}

            model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM"]

            for model_name in model_names:
                auc_score, model, best_params = self.optimize_model(model_name, X_train, y_train, X_test, y_test)
                if auc_score > best_model_score:
                    best_model = model
                    best_model_name = model_name
                    best_model_score = auc_score
                    best_model_params = best_params

            if best_model is None:
                raise CustomException("No valid model was found", sys)

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model: {best_model_name}, AUC: {best_model_score}")

            with mlflow.start_run(run_name=f"{best_model_name}_training"):
                mlflow.log_params(best_model_params)
                mlflow.log_metric("AUC", best_model_score)
                mlflow.sklearn.log_model(best_model, best_model_name.replace(" ", "_"))

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

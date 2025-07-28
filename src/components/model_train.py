import os
import sys
from dataclasses import dataclass
import time
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import dagshub

dagshub.init(repo_owner='Niair', repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")

mlflow.set_experiment("exp_1")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        self.models_config = {
            "RandomForest": {
                "class": RandomForestClassifier,
                "params": {
                    "n_estimators": (100, 300),
                    "max_depth": (3, 15),
                    "min_samples_split": (2, 10),
                    "min_samples_leaf": (1, 10),
                    "n_jobs": -1,
                    "random_state": 42
                }
            },
            "CatBoost": {
                "class": CatBoostClassifier,
                "params": {
                    "iterations": (100, 500),
                    "depth": (4, 10),
                    "learning_rate": (0.01, 0.3, 'log'),
                    "l2_leaf_reg": (1e-3, 10.0, 'log'),
                    "verbose": 0,
                    "random_state": 42
                }
            },
            "XGBoost": {
                "class": XGBClassifier,
                "params": {
                    "n_estimators": (100, 500),
                    "learning_rate": (0.01, 0.3, 'log'),
                    "max_depth": (3, 10),
                    "subsample": (0.6, 1.0),
                    "colsample_bytree": (0.6, 1.0),
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                    "n_jobs": -1,
                    "random_state": 42
                }
            },
            "SVC": {
                "class": SVC,
                "params": {
                    "C": (1e-2, 100, 'log'),
                    "kernel": ["linear", "rbf", "poly", "sigmoid"],
                    "degree": (2, 5),
                    "probability": True,
                    "random_state": 42
                }
            },
            "LogisticRegression": {
                "class": LogisticRegression,
                "params": {
                    "C": (0.01, 10.0),
                    "penalty": ["l1", "l2"],
                    "solver": "saga",
                    "max_iter": 1000,
                    "n_jobs": -1,
                    "random_state": 42
                }
            },
            "LightGBM": {
                "class": LGBMClassifier,
                "params": {
                    "n_estimators": (100, 500),
                    "learning_rate": (0.01, 0.3, 'log'),
                    "num_leaves": (31, 255),
                    "max_depth": (3, 10),
                    "n_jobs": -1,
                    "random_state": 42
                }
            }
        }

    def evaluate_metrics(self, y_true, y_pred, y_probs=None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }
        if y_probs is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
            except:
                metrics["roc_auc"] = 0.0
        return metrics

    def create_model(self, model_name, trial=None, fixed_params=None):
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")

        config = self.models_config[model_name]
        params = {}

        if fixed_params:
            params.update(fixed_params)
            return config["class"](**params)

        if trial is None:
            raise ValueError("Either trial or fixed_params must be provided")

        for param, spec in config["params"].items():
            if isinstance(spec, list):
                params[param] = trial.suggest_categorical(param, spec)
            elif isinstance(spec, tuple):
                if len(spec) == 3 and spec[2] == 'log':
                    params[param] = trial.suggest_float(param, spec[0], spec[1], log=True)
                elif isinstance(spec[0], int):
                    params[param] = trial.suggest_int(param, spec[0], spec[1])
                else:
                    params[param] = trial.suggest_float(param, spec[0], spec[1])
            else:
                params[param] = spec

        return config["class"](**params)

    def objective(self, trial, model_name, X_train, y_train):
        try:
            model = self.create_model(model_name, trial)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
                X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]

                with mlflow.start_run(nested=True, run_name=f"{model_name}_fold_{fold_idx}"):
                    mlflow.set_tag("trial_number", trial.number)
                    start_time = time.time()
                    model.fit(X_fold_train, y_fold_train)
                    fit_time = time.time() - start_time

                    y_val_pred = model.predict(X_fold_val)
                    y_val_probs = model.predict_proba(X_fold_val)[:, 1] if hasattr(model, "predict_proba") else None

                    metrics = self.evaluate_metrics(y_fold_val, y_val_pred, y_val_probs)
                    metrics["fit_time"] = fit_time

                    mlflow.log_params(trial.params)
                    mlflow.log_metrics(metrics)
                    fold_metrics.append(metrics)

            avg_metrics = {
                f"avg_{k}": np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()
            }

            trial.report(avg_metrics.get("avg_roc_auc", 0.0), step=len(fold_metrics))
            return avg_metrics.get("avg_roc_auc", 0.0)

        except Exception as e:
            logging.error(f"Error in objective function for {model_name}: {str(e)}")
            raise CustomException(e, sys)

    def optimize_model(self, model_name, X_train, y_train):
        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
                pruner=MedianPruner()
            )

            study.optimize(lambda trial: self.objective(trial, model_name, X_train, y_train), n_trials=20, n_jobs=1)
            return study.best_params, study.best_value

        except Exception as e:
            logging.error(f"Optimization failed for {model_name}: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            best_model = None
            best_score = -1
            best_metrics = {}
            best_model_name = ""
            best_params_all = {}

            with mlflow.start_run(run_name="classifier_comparison"):
                mlflow.set_tags({"author": "nihal", "project": "customer-churn", "task": "classification"})

                for model_name in self.models_config:
                    try:
                        with mlflow.start_run(run_name=f"{model_name}_optimization", nested=True):
                            logging.info(f"Optimizing {model_name} with Optuna")
                            best_params, best_cv_score = self.optimize_model(model_name, X_train, y_train)

                            model = self.create_model(model_name, fixed_params=best_params)
                            model.fit(X_train, y_train)

                            y_pred = model.predict(X_test)
                            y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                            metrics = self.evaluate_metrics(y_test, y_pred, y_probs)

                            signature = infer_signature(X_test, y_pred)

                            mlflow.log_params(best_params)
                            mlflow.log_metrics(metrics)
                            mlflow.log_metric("cv_roc_auc", best_cv_score)

                            mlflow.sklearn.log_model(
                                sk_model=model,
                                artifact_path=f"{model_name}_model",
                                registered_model_name=f"churn_{model_name}",
                                signature=signature,
                                input_example=X_test[:1],
                                metadata={"cv_score": best_cv_score, "model_type": model_name}
                            )

                            if metrics.get("roc_auc", 0) > best_score:
                                best_score = metrics["roc_auc"]
                                best_model = model
                                best_model_name = model_name
                                best_metrics = metrics
                                best_params_all = best_params

                    except Exception as e:
                        logging.error(f"Failed to optimize {model_name}: {str(e)}")
                        continue

                if best_model is None:
                    raise CustomException("No suitable model found after testing all classifiers", sys)

                save_object(self.model_trainer_config.trained_model_file_path, best_model)

                with mlflow.start_run(run_name="best_overall_model", nested=True):
                    mlflow.set_tags({"best_model": best_model_name, "model_type": "production"})
                    mlflow.log_params(best_params_all)
                    mlflow.log_metrics(best_metrics)
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="production_model",
                        registered_model_name="churn_production_model",
                        signature=infer_signature(X_test, y_pred),
                        input_example=X_test[:1],
                        metadata=best_metrics
                    )
                    mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)

            logging.info(f"Best model: {best_model_name} with ROC AUC: {best_score:.4f}")
            return {
                "model": best_model_name,
                "metrics": best_metrics,
                "params": best_params_all,
                "model_path": self.model_trainer_config.trained_model_file_path,
                "mlflow_run_id": mlflow.active_run().info.run_id
            }

        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise CustomException(e, sys)

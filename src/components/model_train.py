import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
    recall_score, f1_score, make_scorer
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import yaml

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import dagshub

# Initialize DagsHub
dagshub.init(
    repo_owner='Niair',
    repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD',
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")
mlflow.set_experiment("Customer_Churn_Prediction")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    config_path: str = os.path.join("config", "models.yaml")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            with open(self.model_trainer_config.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            raise CustomException(e, sys)

    def _get_proba_scores(self, model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
        """Get probability scores from model"""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # Scale scores to [0, 1] for AUC calculation
            return (scores - scores.min()) / (scores.max() - scores.min())
        else:
            return model.predict(X)

    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Evaluate classification metrics"""
        return {
            "roc_auc": roc_auc_score(y_true, y_probs),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    def _create_model(self, model_name: str, trial: optuna.Trial) -> ClassifierMixin:
        """Create model instance with suggested hyperparameters"""
        params = self.config["model_params"][model_name]
        
        if model_name == "RandomForest":
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", *params["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *params["max_depth"]),
                "min_samples_split": trial.suggest_int("min_samples_split", *params["min_samples_split"]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", *params["min_samples_leaf"]),
                "n_jobs": -1,
                "random_state": 42
            }
            return RandomForestClassifier(**model_params)
        
        elif model_name == "CatBoost":
            model_params = {
                "iterations": trial.suggest_int("iterations", *params["iterations"]),
                "depth": trial.suggest_int("depth", *params["depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *params["learning_rate"], log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *params["l2_leaf_reg"], log=True),
                "verbose": 0,
                "random_state": 42
            }
            return CatBoostClassifier(**model_params)
        
        elif model_name == "XGBoost":
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", *params["n_estimators"]),
                "learning_rate": trial.suggest_float("learning_rate", *params["learning_rate"], log=True),
                "max_depth": trial.suggest_int("max_depth", *params["max_depth"]),
                "subsample": trial.suggest_float("subsample", *params["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *params["colsample_bytree"]),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42
            }
            return XGBClassifier(**model_params)
        
        elif model_name == "SVC":
            model_params = {
                "C": trial.suggest_float("C", *params["C"], log=True),
                "kernel": trial.suggest_categorical("kernel", params["kernel"]),
                "degree": trial.suggest_int("degree", *params["degree"]),
                "probability": True,
                "random_state": 42
            }
            return SVC(**model_params)
        
        elif model_name == "LogisticRegression":
            model_params = {
                "C": trial.suggest_float("C", *params["C"]),
                "penalty": trial.suggest_categorical("penalty", params["penalty"]),
                "solver": "saga" if trial.suggest_categorical("penalty", params["penalty"]) == "l1" else "lbfgs",
                "max_iter": 1000,
                "random_state": 42
            }
            return LogisticRegression(**model_params)
        
        elif model_name == "LightGBM":
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", *params["n_estimators"]),
                "learning_rate": trial.suggest_float("learning_rate", *params["learning_rate"], log=True),
                "num_leaves": trial.suggest_int("num_leaves", *params["num_leaves"]),
                "max_depth": trial.suggest_int("max_depth", *params["max_depth"]),
                "random_state": 42
            }
            return LGBMClassifier(**model_params)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def optimize_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                      n_trials: int = 20) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna"""
        def objective(trial):
            model = self._create_model(model_name, trial)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scorer = make_scorer(roc_auc_score, needs_proba=True)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, scoring=scorer, n_jobs=-1
            )
            return np.mean(scores)
        
        study = optuna.create_study(
            direction="maximize", 
            pruner=MedianPruner(n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value
        }

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, Any]:
        """Train and evaluate multiple models, selecting the best one"""
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            models = self.config["model_params"].keys()
            best_model = None
            best_score = -1
            best_metrics = {}
            best_model_name = ""
            best_params = {}
            
            for model_name in models:
                try:
                    logging.info(f"Optimizing {model_name} with Optuna")
                    start_time = time.time()
                    
                    # Hyperparameter optimization
                    optuna_result = self.optimize_model(model_name, X_train, y_train)
                    params = optuna_result["best_params"]
                    optuna_score = optuna_result["best_value"]
                    
                    # Train final model with best params
                    model = self._create_model(model_name, None)
                    model.set_params(**params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    y_pred = model.predict(X_test)
                    y_probs = self._get_proba_scores(model, X_test)
                    metrics = self.evaluate_metrics(y_test, y_pred, y_probs)
                    test_score = metrics["roc_auc"]
                    
                    # MLflow logging
                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.log_params(params)
                        mlflow.log_metrics(metrics)
                        mlflow.log_metric("optuna_cv_score", optuna_score)
                        mlflow.sklearn.log_model(model, model_name)
                    
                    training_time = time.time() - start_time
                    logging.info(
                        f"{model_name} trained in {training_time:.2f}s. "
                        f"CV AUC: {optuna_score:.4f}, Test AUC: {test_score:.4f}"
                    )
                    
                    # Update best model
                    if test_score > best_score:
                        best_score = test_score
                        best_model = model
                        best_model_name = model_name
                        best_metrics = metrics
                        best_params = params
                        
                except Exception as e:
                    logging.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise CustomException("No valid model was trained", sys)
            
            # Save best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            
            # Log best model separately
            with mlflow.start_run(run_name="Best_Model"):
                mlflow.set_tags({
                    "best_model": best_model_name,
                    "author": "Nihal",
                    "project": "Customer Churn Prediction"
                })
                mlflow.log_params(best_params)
                mlflow.log_metrics(best_metrics)
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)
                signature = infer_signature(X_train, best_model.predict(X_train))
                mlflow.sklearn.log_model(
                    best_model, 
                    "best_model",
                    signature=signature
                )
            
            logging.info(f"Best model: {best_model_name} with AUC: {best_score:.4f}")
            return {
                "best_model": best_model_name,
                "metrics": best_metrics,
                "params": best_params,
                "model_path": self.model_trainer_config.trained_model_file_path
            }
        
        except Exception as e:
            raise CustomException(e, sys)
import os
import sys
from dataclasses import dataclass
import mlflow
import dagshub
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Initialize MLflow tracking
dagshub.init(
    repo_owner='Niair',
    repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD',
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        self.best_models = {}  # Stores best model from each category
        self.global_best = None  # Will store the absolute best model

    def _get_model_from_name(self, model_name, trial=None):
        if trial is None:
            if model_name == "Random Forest":
                return RandomForestClassifier(random_state=42, class_weight='balanced')
            elif model_name == "XGBoost":
                return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif model_name == "LightGBM":
                return LGBMClassifier(random_state=42, class_weight='balanced')
            elif model_name == "CatBoost":
                return CatBoostClassifier(verbose=0, random_state=42)
            elif model_name == "SVM":
                return SVC(probability=True, random_state=42, class_weight='balanced')
            elif model_name == "Logistic Regression":
                return LogisticRegression(random_state=42, class_weight='balanced')
            else:
                raise ValueError("Unsupported model")

        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=42
            )
        elif model_name == "XGBoost":
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                random_state=42
            )
        elif model_name == "LightGBM":
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 60),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                random_state=42,
                class_weight='balanced'
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                verbose=0,
                iterations=trial.suggest_int("iterations", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                depth=trial.suggest_int("depth", 4, 7),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
                random_seed=42
            )
        elif model_name == "SVM":
            return SVC(
                probability=True,
                C=trial.suggest_float("C", 0.1, 10.0, log=True),
                kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
                random_state=42,
                class_weight='balanced'
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 0.1, 5.0),
                max_iter=trial.suggest_int("max_iter", 100, 300),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError("Unsupported model")

    def _run_algorithm_experiment(self, model_name, X_train, y_train, X_test, y_test):
        """Runs complete experiment for one algorithm"""
        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
        else:
            X_train_df = X_train
            X_test_df = X_test

        # Scoring metrics
        scoring = {
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Storage for this algorithm's results
        algorithm_results = {
            "best_model": None,
            "best_auc": -1,
            "best_accuracy": -1,
            "best_params": {},
            "best_metrics": {},
            "trials": []
        }

        def objective(trial):
            with mlflow.start_run(nested=True):
                # Model setup
                model = self._get_model_from_name(model_name, trial)
                pipeline = Pipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('model', model)
                ])

                # Cross-validation
                cv_results = cross_validate(
                    pipeline, X_train_df, y_train,
                    cv=cv_strategy, scoring=scoring,
                    return_train_score=False, n_jobs=-1
                )

                # Calculate mean metrics
                metrics = {
                    "auc": np.mean(cv_results['test_roc_auc']),
                    "accuracy": np.mean(cv_results['test_accuracy']),
                    "precision": np.mean(cv_results['test_precision']),
                    "recall": np.mean(cv_results['test_recall']),
                    "f1": np.mean(cv_results['test_f1'])
                }

                # Store trial results
                trial_data = {
                    "params": trial.params,
                    "cv_metrics": metrics,
                    "test_metrics": None  # Will be filled for best trial
                }
                algorithm_results["trials"].append(trial_data)

                # Log to MLflow
                mlflow.log_params(trial.params)
                mlflow.log_metrics({f"cv_{k}": v for k, v in metrics.items()})

                return metrics["auc"]

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        # Train best model on full data
        best_model = self._get_model_from_name(model_name, trial=optuna.trial.FixedTrial(study.best_params))
        X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_df, y_train)
        best_model.fit(X_train_resampled, y_train_resampled)

        # Evaluate on test set
        y_pred_proba = best_model.predict_proba(X_test_df)[:, 1]
        y_pred = best_model.predict(X_test_df)

        test_metrics = {
            "auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }

        # Update algorithm results
        algorithm_results.update({
            "best_model": best_model,
            "best_auc": test_metrics["auc"],
            "best_accuracy": test_metrics["accuracy"],
            "best_params": study.best_params,
            "best_metrics": test_metrics
        })

        # Update test metrics for best trial
        for trial in algorithm_results["trials"]:
            if trial["params"] == study.best_params:
                trial["test_metrics"] = test_metrics
                break

        # Log final algorithm results
        mlflow.log_metrics({f"best_{k}": v for k, v in test_metrics.items()})
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        
        return algorithm_results

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_names = ["Random Forest", "Logistic Regression", "XGBoost", "LightGBM", "CatBoost", "SVM"]

            # Run each algorithm in its own experiment
            for model_name in model_names:
                with mlflow.start_run(run_name=f"{model_name} Experiment", nested=True):
                    mlflow.set_tag("algorithm", model_name)
                    results = self._run_algorithm_experiment(model_name, X_train, y_train, X_test, y_test)
                    self.best_models[model_name] = results

                    # Save model artifact
                    model_path = os.path.join("artifacts", f"{model_name.replace(' ', '_')}_model.pkl")
                    save_object(model_path, results["best_model"])
                    mlflow.log_artifact(model_path)

            # Compare all algorithms to find global best
            with mlflow.start_run(run_name="Best Models Comparison"):
                global_best_auc = -1
                global_best_model = None
                global_best_name = ""
                
                # Log each algorithm's best performance
                for model_name, results in self.best_models.items():
                    mlflow.log_metrics({
                        f"{model_name}_best_auc": results["best_auc"],
                        f"{model_name}_best_accuracy": results["best_accuracy"]
                    })
                    
                    # Check if this is the new global best
                    if results["best_auc"] > global_best_auc:
                        global_best_auc = results["best_auc"]
                        global_best_model = results["best_model"]
                        global_best_name = model_name
                
                # Log global best
                mlflow.set_tag("global_best_model", global_best_name)
                mlflow.log_metrics({
                    "global_best_auc": global_best_auc,
                    "global_best_accuracy": self.best_models[global_best_name]["best_accuracy"]
                })
                
                # Save global best model
                save_object(self.model_trainer_config.trained_model_file_path, global_best_model)
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)

            return global_best_auc

        except Exception as e:
            raise CustomException(e, sys)
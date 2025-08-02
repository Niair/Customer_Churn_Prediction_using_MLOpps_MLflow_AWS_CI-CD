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

dagshub.init(
    repo_owner='Niair',
    repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD',
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")
mlflow.set_experiment("t2")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

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

    def optimize_model(self, model_name, X_train, y_train, X_test, y_test):
        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
        else:
            X_train_df = X_train
            X_test_df = X_test

        def _roc_auc_proba_scorer(y_true, y_proba):
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            else:
                return roc_auc_score(y_true, y_proba)

        scoring = {
            'roc_auc': make_scorer(_roc_auc_proba_scorer, needs_proba=True),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective(trial):
            with mlflow.start_run(run_name=f"{model_name}_Trial_{trial.number}", nested=True):
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("trial_id", trial.number)

                model = self._get_model_from_name(model_name, trial)
                pipeline = Pipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('model', model)
                ])

                try:
                    cv_results = cross_validate(
                        pipeline, X_train_df, y_train,
                        cv=cv_strategy, scoring=scoring,
                        return_train_score=False, n_jobs=-1
                    )

                    metrics_to_log = {
                        "AUC": np.mean(cv_results['test_roc_auc']),
                        "Precision": np.mean(cv_results['test_precision']),
                        "Recall": np.mean(cv_results['test_recall']),
                        "F1-Score": np.mean(cv_results['test_f1'])
                    }

                    for key in metrics_to_log:
                        if np.isnan(metrics_to_log[key]) or np.isinf(metrics_to_log[key]):
                            logging.warning(f"{model_name} Trial {trial.number} - {key} is NaN/Inf, setting to 0.")
                            metrics_to_log[key] = 0.0

                    prefixed_params = {f"{model_name}_{k}": v for k, v in trial.params.items()}
                    mlflow.log_params(prefixed_params)

                    mlflow.log_metrics({f"{model_name}_CV_{k}": v for k, v in metrics_to_log.items()})

                    # Prune poor trials
                    if metrics_to_log["AUC"] == 0.0:
                        raise optuna.exceptions.TrialPruned()

                    return metrics_to_log["AUC"]

                except Exception as e:
                    logging.error(f"{model_name} Trial {trial.number} failed: {e}")
                    mlflow.set_tag("trial_status", "failed")
                    return -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        best_model = self._get_model_from_name(model_name, trial=optuna.trial.FixedTrial(study.best_params))
        X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_df, y_train)
        best_model.fit(X_train_resampled, y_train_resampled)

        y_pred_proba = best_model.predict_proba(X_test_df)[:, 1]
        y_pred = best_model.predict(X_test_df)

        return (
            roc_auc_score(y_test, y_pred_proba),
            best_model,
            study.best_params,
            {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_pred, zero_division=0)
            }
        )

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            global_best_score = -1.0
            global_best_model = None
            global_best_model_name = None
            global_best_params = {}
            global_best_metrics = {}

            model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM", "Logistic Regression"]

            with mlflow.start_run(run_name="Best_Model_Run"):
                for model_name in model_names:
                    best_score = -1.0
                    best_model = None
                    best_params = {}
                    best_metrics = {}

                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.set_tag("model_name", model_name)

                        try:
                            test_auc, model, params, metrics = self.optimize_model(
                                model_name, X_train, y_train, X_test, y_test
                            )

                            # ✅ Log test scores for the best trial of the model
                            mlflow.log_metrics({
                                f"Best_{model_name.replace(' ', '_')}_AUC": test_auc,
                                f"Best_{model_name.replace(' ', '_')}_Accuracy": metrics["test_accuracy"],
                                f"Best_{model_name.replace(' ', '_')}_Precision": metrics["test_precision"],
                                f"Best_{model_name.replace(' ', '_')}_Recall": metrics["test_recall"],
                                f"Best_{model_name.replace(' ', '_')}_F1": metrics["test_f1"],
                            })

                            # ✅ Save and log the best model artifact of this type
                            model_path = os.path.join("temp_models", f"{model_name.replace(' ', '_')}_model.pkl")
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            save_object(model_path, model)
                            mlflow.log_artifact(model_path, artifact_path="model_artifacts")
                            os.remove(model_path)

                            best_model = model
                            best_score = test_auc
                            best_params = params
                            best_metrics = metrics

                            if test_auc > global_best_score:
                                global_best_model = model
                                global_best_score = test_auc
                                global_best_model_name = model_name
                                global_best_params = params
                                global_best_metrics = metrics

                        except Exception as e:
                            logging.error(f"{model_name} failed: {e}")
                            mlflow.log_param(f"{model_name}_status", "Failed")
                            mlflow.log_metric(f"{model_name}_AUC", -1.0)

                # ✅ Log final global best model after all types
                if global_best_model is None:
                    raise CustomException("All model optimizations failed", sys)

                save_object(self.model_trainer_config.trained_model_file_path, global_best_model)

                mlflow.set_tag("final_model", global_best_model_name)
                mlflow.log_param("Overall_Best_Model", global_best_model_name)
                mlflow.log_metrics({
                    "Overall_Best_AUC": global_best_score,
                    "Overall_Best_Accuracy": global_best_metrics.get("test_accuracy", 0.0),
                    "Overall_Best_Precision": global_best_metrics.get("test_precision", 0.0),
                    "Overall_Best_Recall": global_best_metrics.get("test_recall", 0.0),
                    "Overall_Best_F1": global_best_metrics.get("test_f1", 0.0)
                })
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path, artifact_path="final_best_model")

            logging.info(f"Best overall model: {global_best_model_name} with AUC: {global_best_score}")
            return global_best_score

        except Exception as e:
            raise CustomException(e, sys)

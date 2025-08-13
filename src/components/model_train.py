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
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imblearn_Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class DummyContextManager:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class ModelTrainer:
    def __init__(self, enable_logging=True):
        self.model_trainer_config = ModelTrainerConfig()
        self.enable_logging = enable_logging
        if self.enable_logging and os.environ.get("CI") != "true":
            dagshub.init(
                repo_owner='Niair',
                repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD',
                mlflow=True
            )
            mlflow.set_tracking_uri(
                "https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow"
            )


    # ---------- SANITIZATION HELPERS ----------
    def _sanitize_params(self, params: dict):
        clean = {}
        for k, v in params.items():
            key = str(k).replace(" ", "_").replace("-", "_")
            if isinstance(v, (np.generic, np.ndarray)):
                v = v.item()
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = "null"
            clean[key] = str(v)
        return clean

    def _sanitize_metrics(self, metrics: dict):
        clean = {}
        for k, v in metrics.items():
            key = str(k).replace(" ", "_").replace("-", "_")
            if isinstance(v, (np.generic, np.ndarray)):
                v = v.item()
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = 0.0
            clean[key] = float(v)
        return clean

    # ---------- MLFLOW HELPERS ----------
    def _mlflow_start_run(self, run_name=None, **kwargs):
        if run_name:
            run_name = run_name.strip().replace(" ", "_")
        if self.enable_logging:
            return mlflow.start_run(run_name=run_name, **kwargs)
        return DummyContextManager()

    def _mlflow_log_params(self, params):
        if self.enable_logging:
            mlflow.log_params(params)

    def _mlflow_log_metrics(self, metrics):
        if self.enable_logging:
            mlflow.log_metrics(metrics)

    def _mlflow_log_artifact(self, local_path, artifact_path=None):
        if self.enable_logging:
            mlflow.log_artifact(local_path, artifact_path)

    def _mlflow_set_tag(self, key, value):
        if self.enable_logging:
            mlflow.set_tag(key, value)

    def _mlflow_set_experiment(self, experiment_name):
        clean_name = experiment_name.strip().replace(" ", "_")
        if self.enable_logging:
            try:
                mlflow.set_experiment(clean_name)
            except Exception as e:
                logging.error(f"Failed to set experiment {clean_name}: {e}")
                raise

    # ---------- MODEL FACTORY ----------
    def _get_model_from_name(self, model_name, trial=None):
        if trial is None:
            if model_name == "Random Forest":
                return RandomForestClassifier(random_state=42, class_weight='balanced')
            elif model_name == "XGBoost":
                return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif model_name == "LightGBM":
                return LGBMClassifier(random_state=42)
            elif model_name == "CatBoost":
                return CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced')
            elif model_name == "SVM":
                return SVC(probability=True, random_state=42, class_weight='balanced')
            elif model_name == "Logistic Regression":
                return LogisticRegression(random_state=42, class_weight='balanced')
            else:
                raise ValueError("Unsupported model")
        # With Optuna params
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                class_weight='balanced', random_state=42
            )
        elif model_name == "XGBoost":
            return XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
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
                random_state=42
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                verbose=0,
                iterations=trial.suggest_int("iterations", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                depth=trial.suggest_int("depth", 4, 7),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
                auto_class_weights='Balanced', random_seed=42
            )
        elif model_name == "SVM":
            return SVC(
                probability=True,
                C=trial.suggest_float("C", 0.1, 10.0, log=True),
                kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
                class_weight='balanced', random_state=42
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 0.1, 5.0),
                max_iter=trial.suggest_int("max_iter", 100, 300),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                class_weight='balanced', random_state=42
            )

    # ---------- OPTIMIZATION ----------
    def optimize_model(self, model_name, X_train, y_train, X_test, y_test, n_trials=30, experiment_name="churn_prediction_experiment_main"):
        self._mlflow_set_experiment(experiment_name)

        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
        else:
            X_train_df, X_test_df = X_train, X_test

        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if n_pos < 2:
            logging.error(f"Insufficient positive samples ({n_pos}) for {model_name}")
            return 0.0, None, {}, {}

        def _roc_auc_proba_scorer(y_true, estimator, X):
            """Robust ROC AUC scorer supporting predict_proba and decision_function."""
            try:
                if hasattr(estimator, "predict_proba"):
                    y_proba = estimator.predict_proba(X)
                    if y_proba.ndim == 1:  # Single column
                        return roc_auc_score(y_true, y_proba)
                    elif y_proba.shape[1] == 2:
                        return roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        return roc_auc_score(y_true, y_proba, multi_class='ovr')
                elif hasattr(estimator, "decision_function"):
                    scores = estimator.decision_function(X)
                    return roc_auc_score(y_true, scores)
                else:
                    logging.error("Model has neither predict_proba nor decision_function")
                    return 0.0
            except Exception as e:
                logging.error(f"AUC calculation failed: {e}")
                return 0.0


        scoring = {
            'roc_auc': make_scorer(
                lambda y_true, y_pred, **kwargs: _roc_auc_proba_scorer(
                    y_true, kwargs.get('estimator'), kwargs.get('X')
                ),
                needs_proba=False
            ),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                model = self._get_model_from_name(model_name, trial)
                if model_name in ["XGBoost", "LightGBM"]:
                    model.set_params(scale_pos_weight=scale_pos_weight)

                pipeline = imblearn_Pipeline([
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
                        "Accuracy": np.mean(cv_results['test_accuracy']),
                        "Precision": np.mean(cv_results['test_precision']),
                        "Recall": np.mean(cv_results['test_recall']),
                        "F1_Score": np.mean(cv_results['test_f1'])
                    }
                    mlflow.log_params(self._sanitize_params(trial.params))
                    mlflow.log_metrics(self._sanitize_metrics(metrics_to_log))

                    return metrics_to_log["AUC"]

                except Exception as e:
                    logging.error(f"Error during trial {trial.number}: {e}")
                    return -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_model = self._get_model_from_name(model_name, trial=optuna.trial.FixedTrial(study.best_params))
        if model_name in ["XGBoost", "LightGBM"]:
            best_model.set_params(scale_pos_weight=scale_pos_weight)

        best_pipeline = imblearn_Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', best_model)
        ])
        best_pipeline.fit(X_train_df, y_train)

        y_pred = best_pipeline.predict(X_test_df)
        y_proba = best_pipeline.predict_proba(X_test_df)[:, 1]

        test_metrics = {
            "Best_AUC": roc_auc_score(y_test, y_proba),
            "Best_Accuracy": accuracy_score(y_test, y_pred),
            "Best_Precision": precision_score(y_test, y_pred, zero_division=0),
            "Best_Recall": recall_score(y_test, y_pred, zero_division=0),
            "Best_F1": f1_score(y_test, y_pred, zero_division=0)
        }

        if self.enable_logging:
            mlflow.log_metrics(test_metrics)
            mlflow.log_params(study.best_params)

        return test_metrics["Best_AUC"], best_pipeline, study.best_params, test_metrics

    # ---------- MAIN TRAINER ----------
    def initiate_model_trainer(self, train_arr, test_arr, n_trials=30, experiment_name="churn_prediction_experiment_main", model_names=None):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            best_overall_score = -1
            best_overall_model = None
            best_model_name = None
            best_params = {}
            best_metrics = {}

            if model_names is None:
                model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM", "Logistic Regression"]

            experiment_name = "churn_prediction_experiment_main"
            self._mlflow_set_experiment(experiment_name)

            with self._mlflow_start_run(run_name="Best_Model_Run"):
                for model_name in model_names:
                    safe_model_name = model_name.replace(" ", "_")
                    with self._mlflow_start_run(run_name=safe_model_name, nested=True):
                        self._mlflow_set_tag("model_name", model_name)
                        try:
                            test_auc, model, params, metrics = self.optimize_model(
                                model_name, X_train, y_train, X_test, y_test, n_trials=n_trials, experiment_name=experiment_name
                            )

                            self._mlflow_log_metrics(self._sanitize_metrics(metrics))
                            self._mlflow_log_params(self._sanitize_params(params))

                            for k, v in metrics.items():
                                mlflow.set_tag(f"best_{k}", round(v, 4))
                            for k, v in params.items():
                                mlflow.set_tag(f"param_{k}", v)

                            # Track best overall model
                            if test_auc > best_overall_score:
                                best_overall_score = test_auc
                                best_overall_model = model
                                best_model_name = model_name
                                best_params = params
                                best_metrics = metrics
                                
                        except Exception as e:
                            logging.error(f"{model_name} failed: {e}")

                mlflow.set_tag("best_model_name", best_model_name)
                
                mlflow.log_params(best_params)
                mlflow.log_metrics(best_metrics)

                if best_overall_model is None:
                    raise CustomException("No model trained successfully", sys)

                save_object(self.model_trainer_config.trained_model_file_path, best_overall_model)
                self._mlflow_log_params(self._sanitize_params({"Best_Model": best_model_name}))
                self._mlflow_log_metrics(self._sanitize_metrics(best_metrics))

                logging.info(f"Best model: {best_model_name} with AUC: {best_overall_score}")
                return best_overall_score

        except Exception as e:
            raise CustomException(e, sys)

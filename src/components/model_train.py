import os
import sys
from dataclasses import dataclass

import mlflow
import dagshub
import optuna

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from functools import partial

from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

dagshub.init(repo_owner='Niair', repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")
mlflow.set_experiment("ex1")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def _get_model_from_name(self, model_name, trial=None):
        if trial is None:
            if model_name == "Random Forest":
                return RandomForestClassifier(random_state=42, class_weight="balanced")  # UPDATED
            elif model_name == "XGBoost":
                return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif model_name == "LightGBM":
                return LGBMClassifier(random_state=42)
            elif model_name == "CatBoost":
                return CatBoostClassifier(verbose=0, random_state=42)
            elif model_name == "SVM":
                return SVC(probability=True, random_state=42, class_weight="balanced") 
            elif model_name == "Logistic Regression":
                return LogisticRegression(random_state=42, class_weight="balanced") 
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=42,
                class_weight="balanced"  # UPDATED
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
                random_state=42
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
                class_weight="balanced"  # UPDATED
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 0.1, 5.0),
                max_iter=trial.suggest_int("max_iter", 100, 300),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                random_state=42,
                class_weight="balanced"  # UPDATED
            )
        else:
            raise ValueError(f"Unsupported model name for optimization: {model_name}")

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test arrays into features (X) and target (y).")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("Applying SMOTE to handle class imbalance.")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            best_overall_model_score = -1.0
            best_overall_model = None
            best_overall_model_name = None
            best_overall_model_params = {}
            best_overall_additional_metrics = {}

            model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM", "Logistic Regression"]

            with mlflow.start_run(run_name="Best_Model_Run") as parent_run:
                for model_name in model_names:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        try:
                            if len(np.unique(y_train)) < 2:
                                logging.warning(
                                    f"Skipping {model_name}: y_train has only one class.")
                                mlflow.log_param(f"{model_name}_status", "Skipped - Single Class")
                                mlflow.log_metric(f"{model_name}_Test_AUC", 0.0)
                                continue

                            test_auc, model, best_params, additional_metrics = self.optimize_model(
                                model_name, X_train, y_train, X_test, y_test
                            )

                            mlflow.log_params(best_params)
                            mlflow.log_metric("Test_AUC", test_auc)
                            mlflow.log_metric("Test_Accuracy", additional_metrics["test_accuracy"])
                            mlflow.log_metric("Test_Precision", additional_metrics["test_precision"])
                            mlflow.log_metric("Test_Recall", additional_metrics["test_recall"])
                            mlflow.log_metric("Test_F1-Score", additional_metrics["test_f1"])
                            mlflow.log_metric("Optuna_CV_Best_AUC", additional_metrics["optuna_cv_best_auc"])

                            model_path = os.path.join("temp_models", f"{model_name.replace(' ', '_')}_model.pkl")
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            save_object(file_path=model_path, obj=model)
                            mlflow.log_artifact(local_path=model_path, artifact_path="model_artifacts")
                            logging.info(f"Logged {model_name} model.")
                            os.remove(model_path)

                            if not np.isnan(test_auc) and test_auc > best_overall_model_score:
                                best_overall_model = model
                                best_overall_model_score = test_auc
                                best_overall_model_name = model_name
                                best_overall_model_params = best_params
                                best_overall_additional_metrics = additional_metrics
                        except Exception as model_optim_e:
                            logging.error(
                                f"Optimization for {model_name} failed: {model_optim_e}",
                                exc_info=True
                            )
                            mlflow.log_param(f"{model_name}_status", "Failed Optimization")
                            mlflow.log_metric(f"{model_name}_Test_AUC", -1.0)

                if best_overall_model is None:
                    logging.warning("No overall best model found.")
                    return -1.0

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_overall_model
                )

                mlflow.log_param("Overall_Best_Model", best_overall_model_name)
                mlflow.log_metrics({
                    "Overall_Best_Test_AUC": best_overall_model_score,
                    "Overall_Best_Test_Accuracy": best_overall_additional_metrics.get("test_accuracy", 0.0),
                    "Overall_Best_Test_Precision": best_overall_additional_metrics.get("test_precision", 0.0),
                    "Overall_Best_Test_Recall": best_overall_additional_metrics.get("test_recall", 0.0),
                    "Overall_Best_Test_F1-Score": best_overall_additional_metrics.get("test_f1", 0.0),
                    "Overall_Best_Optuna_CV_AUC": best_overall_additional_metrics.get("optuna_cv_best_auc", 0.0)
                })
                mlflow.log_artifact(local_path=self.model_trainer_config.trained_model_file_path, artifact_path="final_best_model")

            logging.info(f"Best overall model: {best_overall_model_name} with Test AUC: {best_overall_model_score}")
            return best_overall_model_score

        except Exception as e:
            raise CustomException(e, sys)

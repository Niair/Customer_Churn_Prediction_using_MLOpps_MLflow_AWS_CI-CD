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
from imblearn.pipeline import Pipeline as imblearn_Pipeline  # it is done like this to avoids name conflict
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

dagshub.init(repo_owner='Niair', repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")
mlflow.set_experiment("ex2")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
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
        
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                class_weight='balanced', 
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
                random_state=42
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                verbose=0,
                iterations=trial.suggest_int("iterations", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                depth=trial.suggest_int("depth", 4, 7),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
                auto_class_weights='Balanced', 
                random_seed=42
            )
        elif model_name == "SVM":
            return SVC(
                probability=True,
                C=trial.suggest_float("C", 0.1, 10.0, log=True),
                kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
                class_weight='balanced', 
                random_state=42
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 0.1, 5.0),
                max_iter=trial.suggest_int("max_iter", 100, 300),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                class_weight='balanced', 
                random_state=42
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

        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        def _roc_auc_proba_scorer(y_true, y_proba):
            try:
                # Handle binary and multi-class cases
                if y_proba.ndim == 1:
                    return roc_auc_score(y_true, y_proba)
                elif y_proba.shape[1] == 2:
                    return roc_auc_score(y_true, y_proba[:, 1])
                else:
                    return roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                logging.error(f"AUC calculation failed: {e}")
                return 0.0

        scoring = {
            'roc_auc': make_scorer(_roc_auc_proba_scorer, needs_proba=True),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        def objective(trial):
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                model = self._get_model_from_name(model_name, trial)
                
                if model_name == "XGBoost":
                    model.set_params(scale_pos_weight=scale_pos_weight)
                elif model_name == "LightGBM":
                    model.set_params(scale_pos_weight=scale_pos_weight)
                
                pipeline = imblearn_Pipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('model', model)
                ])
                
                try:
                    cv_results = cross_validate(
                        pipeline, X_train_df, y_train,
                        cv=cv_strategy,
                        scoring=scoring,
                        return_train_score=False,
                        n_jobs=-1
                    )
                    mean_auc = np.mean(cv_results['test_roc_auc'])
                    mean_accuracy = np.mean(cv_results['test_accuracy'])
                    mean_precision = np.mean(cv_results['test_precision'])
                    mean_recall = np.mean(cv_results['test_recall'])
                    mean_f1 = np.mean(cv_results['test_f1'])
                    
                    metrics_to_log = {
                        "AUC": mean_auc,
                        "Accuracy": mean_accuracy,
                        "Precision": mean_precision,
                        "Recall": mean_recall,
                        "F1-Score": mean_f1
                    }
                    for metric_name, value in metrics_to_log.items():
                        if np.isnan(value) or np.isinf(value):
                            logging.warning(f"Trial {trial.number} for {model_name}: {metric_name} is NaN/Inf. Setting to 0.0.")
                            metrics_to_log[metric_name] = 0.0
                    
                    if np.isnan(metrics_to_log["AUC"]) or np.isinf(metrics_to_log["AUC"]):
                         logging.warning(f"Trial {trial.number} for {model_name}: AUC is NaN/Inf. Pruning.")
                         raise optuna.exceptions.TrialPruned()
                    
                    mlflow.log_params(trial.params)
                    mlflow.log_metrics(metrics_to_log)
                    return metrics_to_log["AUC"]
                
                except Exception as e:
                    logging.error(f"Error during trial {trial.number} for {model_name}: {e}")
                    return -1.0

        study = optuna.create_study(direction="maximize")
        try:
            study.optimize(objective, n_trials=3, n_jobs=1)
        except Exception as e:
            logging.error(f"Optuna study failed: {e}")
            raise e
        
        if not study.trials:
            raise ValueError("No trials completed successfully.")
        
        best_model = self._get_model_from_name(model_name, trial=optuna.trial.FixedTrial(study.best_params))
        
        if model_name == "XGBoost":
            best_model.set_params(scale_pos_weight=scale_pos_weight)
        elif model_name == "LightGBM":
            best_model.set_params(scale_pos_weight=scale_pos_weight)
        
        best_pipeline = imblearn_Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', best_model)
        ])
        
        best_pipeline.fit(X_train_df, y_train)
        
        if hasattr(best_pipeline, 'predict_proba'):
            y_pred_proba = best_pipeline.predict_proba(X_test_df)[:, 1]
        else:
            y_pred_proba = best_pipeline.decision_function(X_test_df)
        
        y_pred = best_pipeline.predict(X_test_df)
        
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        
        return test_auc, best_pipeline, study.best_params, {
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
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
                            test_auc, model, best_params, additional_metrics = self.optimize_model(
                                model_name, X_train, y_train, X_test, y_test
                            )
                            
                            mlflow.log_params(best_params)
                            mlflow.log_metric("Best_AUC", test_auc)
                            mlflow.log_metric("Best_Accuracy", additional_metrics["test_accuracy"])
                            mlflow.log_metric("Best_Precision", additional_metrics["test_precision"])
                            mlflow.log_metric("Best_Recall", additional_metrics["test_recall"])
                            mlflow.log_metric("Best_F1-Score", additional_metrics["test_f1"])
                            
                            model_path = os.path.join("temp_models", f"{model_name.replace(' ', '_')}_model.pkl")
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            save_object(file_path=model_path, obj=model)
                            mlflow.log_artifact(local_path=model_path, artifact_path="model_artifacts")
                            logging.info(f"Logged {model_name} model as artifact.")
                            os.remove(model_path)
                            
                            if test_auc > best_overall_model_score:
                                best_overall_model = model
                                best_overall_model_score = test_auc
                                best_overall_model_name = model_name
                                best_overall_model_params = best_params
                                best_overall_additional_metrics = additional_metrics
                                
                        except Exception as model_optim_e:
                            logging.error(f"Model {model_name} failed: {model_optim_e}")
                            mlflow.log_param(f"{model_name}_status", "Failed")
                
                if best_overall_model is None:
                    logging.error("All models failed training")
                    return -1.0
                
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_overall_model
                )
                
                mlflow.log_param("Overall_Best_Model", best_overall_model_name)
                mlflow.log_metrics({
                    "Overall_Best_AUC": best_overall_model_score,
                    "Overall_Best_Accuracy": best_overall_additional_metrics["test_accuracy"],
                    "Overall_Best_Precision": best_overall_additional_metrics["test_precision"],
                    "Overall_Best_Recall": best_overall_additional_metrics["test_recall"],
                    "Overall_Best_F1-Score": best_overall_additional_metrics["test_f1"]
                })
                mlflow.log_artifact(
                    local_path=self.model_trainer_config.trained_model_file_path,
                    artifact_path="final_best_model"
                )
            
            logging.info(f"Best model: {best_overall_model_name} with AUC: {best_overall_model_score}")
            return best_overall_model_score
        
        except Exception as e:
            raise CustomException(e, sys)
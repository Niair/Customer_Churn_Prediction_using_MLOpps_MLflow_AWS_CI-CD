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

from src.exception import CustomException
from src.logger import logging

# Assuming save_object is correctly implemented in src.utils
from src.utils import save_object

# Define a small epsilon to avoid absolute zero for metrics if needed
EPSILON = 1e-9

# --- MLflow Initialization Block ---
# Flag to indicate if MLflow is successfully initialized
MLFLOW_INITIALIZED = False
try:
    dagshub.init(repo_owner='Niair', repo_name='Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Niair/Customer_Churn_Prediction_using_MLOpps_MLflow_AWS_CI-CD.mlflow")

    # Attempt to set experiment, with a specific experiment name
    # Consider changing the name if 'churn_model_optimization_v1' still gives issues
    EXPERIMENT_NAME = "churn_model_optimization_v2"
    mlflow.set_experiment(EXPERIMENT_NAME)

    logging.info(f"MLflow tracking initialized for Dagshub. Experiment: '{EXPERIMENT_NAME}'")
    MLFLOW_INITIALIZED = True
except Exception as e:
    logging.error(f"FATAL ERROR: Failed to initialize MLflow/Dagshub tracking: {e}. All MLflow operations will be skipped.", exc_info=True)
    MLFLOW_INITIALIZED = False # Ensure this is False if init fails


@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def _get_model_from_name(self, model_name, trial=None):
        """
        Returns a model instance, either with default parameters (if trial is None)
        or with Optuna-suggested parameters.
        Includes random_state for reproducibility.
        """
        if trial is None:
            # Default models for final training or if not in Optuna context
            if model_name == "Random Forest":
                return RandomForestClassifier(random_state=42)
            elif model_name == "XGBoost":
                return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            elif model_name == "LightGBM":
                return LGBMClassifier(random_state=42)
            elif model_name == "CatBoost":
                return CatBoostClassifier(verbose=0, random_state=42)
            elif model_name == "SVM":
                return SVC(probability=True, random_state=42)
            elif model_name == "Logistic Regression":
                return LogisticRegression(random_state=42)
            else:
                raise ValueError(f"Unsupported model for default instantiation: {model_name}")

        # Optuna-tuned models with refined hyperparameter ranges
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
                random_state=42
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                verbose=0,
                iterations=trial.suggest_int("iterations", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
                depth=trial.suggest_int("depth", 4, 7),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 5.0, log=True),
                random_seed=42 # CatBoost uses random_seed
            )
        elif model_name == "SVM":
            return SVC(
                probability=True,
                C=trial.suggest_float("C", 0.1, 10.0, log=True),
                kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
                random_state=42
            )
        elif model_name == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 0.1, 5.0),
                max_iter=trial.suggest_int("max_iter", 100, 300),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")

    def optimize_model(self, model_name, X_train, y_train, X_test, y_test):
        """
        Optimizes a given model using Optuna with cross-validation.
        Logs metrics for each trial and the best model for the type to MLflow.
        """
        # Convert X_train/X_test to DataFrame if they are numpy arrays
        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
        else:
            X_train_df = X_train
            X_test_df = X_test

        # --- Custom Scorer for ROC AUC to handle needs_proba TypeError ---
        def _roc_auc_proba_scorer(y_true, y_proba):
            """
            Custom scorer for ROC AUC that handles y_proba[:, 1] and potential inversion.
            `make_scorer(needs_proba=True)` will pass y_proba (2D array) to this function.
            """
            logging.info(f"DEBUG: _roc_auc_proba_scorer - y_true shape: {y_true.shape}, y_proba shape: {y_proba.shape}")
            # Print a sample of y_true and y_proba (first few elements)
            # Ensure y_proba is not empty before attempting to slice
            y_proba_sample = y_proba[:5, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 and y_proba.shape[0] > 0 else (y_proba[:5] if y_proba.shape[0] > 0 else [])
            logging.info(f"DEBUG: _roc_auc_proba_scorer - y_true[:5]: {y_true[:5]}, y_proba_relevant_slice[:5]: {y_proba_sample}")


            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                try:
                    # Check for uniform labels or predictions, which can cause AUC to be NaN
                    if len(np.unique(y_true)) < 2 or len(np.unique(y_proba[:, 1])) < 2:
                        logging.warning(f"DEBUG: _roc_auc_proba_scorer - Skipping AUC calculation due to uniform labels or predictions in fold. y_true unique: {np.unique(y_true)}, y_proba[:,1] unique: {np.unique(y_proba[:,1])}")
                        return 0.0 # Return 0.0 if AUC cannot be meaningfully calculated

                    current_auc = roc_auc_score(y_true, y_proba[:, 1])
                    logging.info(f"DEBUG: _roc_auc_proba_scorer - calculated current_auc: {current_auc}")
                    # Heuristic to detect and fix inverted predictions (AUC near 0 with good accuracy)
                    if current_auc < 0.5 and current_auc >= 0: # Note: AUC can be exactly 0.5 for random
                        # Log if MLflow is active, otherwise just warn
                        if MLFLOW_INITIALIZED and mlflow.active_run():
                             logging.warning(f"Trial {mlflow.active_run().info.run_name}: AUC ({current_auc:.4f}) is below 0.5. Flipping probabilities.")
                        else:
                             logging.warning(f"AUC ({current_auc:.4f}) is below 0.5. Flipping probabilities (MLflow not active).")
                        flipped_auc = roc_auc_score(y_true, 1 - y_proba[:, 1]) # Flip probabilities
                        logging.info(f"DEBUG: _roc_auc_proba_scorer - flipped_auc: {flipped_auc}")
                        return flipped_auc
                    else:
                        return current_auc
                except ValueError as ve:
                    # This can happen if y_true contains only one class, or y_proba is degenerate
                    logging.error(f"DEBUG: _roc_auc_proba_scorer - ValueError calculating AUC for standard shape: {ve}. y_true/y_proba might be problematic. Returning 0.0.", exc_info=True)
                    return 0.0 # Return 0.0 or a sensible default if AUC calculation fails
                except Exception as ex:
                    logging.error(f"DEBUG: _roc_auc_proba_scorer - Unexpected error calculating AUC: {ex}. Returning 0.0.", exc_info=True)
                    return 0.0
            else:
                if MLFLOW_INITIALIZED and mlflow.active_run():
                    logging.warning(f"Trial {mlflow.active_run().info.run_name}: roc_auc_score received unexpected y_proba shape ({y_proba.shape}). Using direct y_proba.")
                else:
                    logging.warning(f"roc_auc_score received unexpected y_proba shape ({y_proba.shape}). Using direct y_proba (MLflow not active).")
                try:
                    # Check for uniform labels or predictions for the direct y_proba case too
                    if len(np.unique(y_true)) < 2 or len(np.unique(y_proba)) < 2:
                        logging.warning(f"DEBUG: _roc_auc_proba_scorer - Skipping AUC calculation due to uniform labels or predictions in fold for direct y_proba. y_true unique: {np.unique(y_true)}, y_proba unique: {np.unique(y_proba)}")
                        return 0.0 # Return 0.0 if AUC cannot be meaningfully calculated

                    return roc_auc_score(y_true, y_proba)
                except ValueError as ve:
                    logging.error(f"DEBUG: _roc_auc_proba_scorer - ValueError calculating AUC for unexpected shape: {ve}. Returning 0.0.", exc_info=True)
                    return 0.0
                except Exception as ex:
                    logging.error(f"DEBUG: _roc_auc_proba_scorer - Unexpected error calculating AUC for unexpected shape: {ex}. Returning 0.0.", exc_info=True)
                    return 0.0


        # Define scoring metrics for cross_validate
        scoring = {
            'roc_auc': make_scorer(_roc_auc_proba_scorer, needs_proba=True), # Use the custom scorer
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        # Use StratifiedKFold for classification to maintain class balance across folds
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            """
            Optuna objective function for hyperparameter optimization with cross-validation.
            Logs trial-specific metrics and parameters to MLflow.
            """
            # Only start MLflow run if tracking is initialized
            if MLFLOW_INITIALIZED:
                with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                    _run_mlflow_logging = True
            else:
                _run_mlflow_logging = False # Skip MLflow logging for this trial

            model = self._get_model_from_name(model_name, trial)

            try:
                # Perform cross-validation
                cv_results = cross_validate(
                    model, X_train_df, y_train, # Use DataFrame for training
                    cv=cv_strategy,
                    scoring=scoring,
                    return_train_score=False, # We only need test scores for evaluation
                    n_jobs=-1 # Use all available cores for CV
                )

                # Calculate mean scores across folds
                # Ensure values extracted are floats, or converted to 0.0 if lists are empty
                mean_auc = float(np.mean(cv_results['test_roc_auc'])) if len(cv_results['test_roc_auc']) > 0 else 0.0
                mean_accuracy = float(np.mean(cv_results['test_accuracy'])) if len(cv_results['test_accuracy']) > 0 else 0.0
                mean_precision = float(np.mean(cv_results['test_precision'])) if len(cv_results['test_precision']) > 0 else 0.0
                mean_recall = float(np.mean(cv_results['test_recall'])) if len(cv_results['test_recall']) > 0 else 0.0
                mean_f1 = float(np.mean(cv_results['test_f1'])) if len(cv_results['test_f1']) > 0 else 0.0

                logging.info(f"DEBUG: Trial {trial.number} raw mean_auc: {mean_auc}, mean_accuracy: {mean_accuracy}")

                # --- Robust NaN/Inf check and sanitization for all metrics ---
                metrics_to_log = {
                    "AUC": mean_auc,
                    "Accuracy": mean_accuracy,
                    "Precision": mean_precision,
                    "Recall": mean_recall,
                    "F1-Score": mean_f1
                }

                logging.info(f"DEBUG: Trial {trial.number} metrics_to_log (before sanitization): {metrics_to_log}")

                sanitized_metrics = {}
                for metric_name, value in metrics_to_log.items():
                    # Use np.nan_to_num for robustness: nan -> 0, inf -> large float
                    sanitized_value = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
                    # Ensure non-AUC metrics are not negative
                    if metric_name != "AUC" and sanitized_value < 0:
                        sanitized_value = 0.0
                    sanitized_metrics[metric_name] = sanitized_value

                logging.info(f"DEBUG: Trial {trial.number} sanitized_metrics (after sanitization): {sanitized_metrics}")

                if _run_mlflow_logging:
                    # Log parameters to the current nested MLflow trial run
                    mlflow.log_params(trial.params)
                    # Log mean metrics from cross-validation to the current nested MLflow trial run
                    mlflow.log_metrics(sanitized_metrics) # Use sanitized metrics

                # Optuna objective typically returns the primary metric to optimize
                # Prune if the primary metric (AUC) is still effectively invalid after handling
                if sanitized_metrics["AUC"] <= EPSILON: # Check for very low or zero AUC
                    logging.warning(f"Trial {trial.number} for {model_name}: Primary optimization metric (AUC) is too low ({sanitized_metrics['AUC']:.4f}). Pruning this trial.")
                    raise optuna.exceptions.TrialPruned()

                return sanitized_metrics["AUC"] # Return the (possibly adjusted and sanitized) AUC for Optuna

            except Exception as e:
                # Log the error for the trial but return a very low value for Optuna to continue
                logging.error(f"Error during trial {trial.number} for {model_name}: {e}", exc_info=True)
                # Return a score that Optuna will minimize when maximizing, ensuring it's a valid float
                return float(0.0) # Using 0.0 directly as a known valid float

        study = optuna.create_study(direction="maximize")
        try:
            # Optimize the objective function. n_jobs=1 for initial stability.
            study.optimize(objective, n_trials=10, n_jobs=1)
        except Exception as e:
            logging.error(f"Optuna study optimization for {model_name} failed unexpectedly: {e}", exc_info=True)
            # Do not re-raise here. Allow the outer loop to continue trying other models.
            # This exception will be caught by the outer try-except for `model_name` loop.
            return float(0.0), None, {}, {"test_accuracy": float(0.0), "test_precision": float(0.0), "test_recall": float(0.0), "test_f1": float(0.0)}


        # Check if any trials completed successfully before accessing best_params/best_value
        successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not successful_trials:
            logging.warning(f"No trials were completed successfully for model {model_name}. All trials might have failed, been pruned, or returned invalid AUCs.")
            # Return placeholder values to avoid crashing the outer loop
            return float(0.0), None, {}, {"test_accuracy": float(0.0), "test_precision": float(0.0), "test_recall": float(0.0), "test_f1": float(0.0)}

        # If best_value is 0.0 or negative, it means all successful trials returned 0.0 or bad
        if study.best_value <= EPSILON: # Check if best value is still effectively 0 or negative
            logging.warning(f"For {model_name}, the best trial's AUC is very low or negative ({study.best_value:.4f}). This model type might not be suitable or needs further hyperparameter tuning.")
            # If all are bad, return default bad values
            return float(0.0), None, {}, {"test_accuracy": float(0.0), "test_precision": float(0.0), "test_recall": float(0.0), "test_f1": float(0.0)}

        logging.info(f"Best params for {model_name}: {study.best_params}")
        logging.info(f"Best AUC for {model_name}: {study.best_value}")

        # Re-train the best model on the full training data (X_train, y_train) using best params
        best_model_for_type = self._get_model_from_name(model_name, trial=optuna.trial.FixedTrial(study.best_params))
        best_model_for_type.fit(X_train_df, y_train)

        # Evaluate this best model on the hold-out test set
        y_pred_proba_raw = None
        if hasattr(best_model_for_type, 'predict_proba'):
            y_pred_proba_raw = best_model_for_type.predict_proba(X_test_df)[:, 1]
        elif hasattr(best_model_for_type, 'decision_function'):
             y_pred_proba_raw = best_model_for_type.decision_function(X_test_df)
        else:
            logging.warning(f"Model {model_name} does not have predict_proba or decision_function. AUC will not be computed.")
            # If AUC cannot be computed, set to a known bad value.
            test_auc = 0.0
            y_pred = best_model_for_type.predict(X_test_df) # Still get class predictions

        if y_pred_proba_raw is not None:
            # Ensure y_pred_proba_raw is a numpy array of floats
            y_pred_proba_raw = np.array(y_pred_proba_raw).astype(float)

            # Check for uniform labels or predictions for final test AUC
            if len(np.unique(y_test)) < 2 or len(np.unique(y_pred_proba_raw)) < 2:
                logging.warning(f"Final test AUC for {model_name}: Skipping AUC calculation due to uniform labels or predictions. Returning 0.0.")
                test_auc = 0.0
            else:
                # --- Apply the inversion check for the final test_auc ---
                temp_auc_check_final = roc_auc_score(y_test, y_pred_proba_raw)
                if temp_auc_check_final < 0.5 and temp_auc_check_final >= 0:
                    test_auc = roc_auc_score(y_test, 1 - y_pred_proba_raw)
                    logging.info(f"Final test AUC for {model_name} was {temp_auc_check_final:.4f}, indicating inversion. Flipped to {test_auc:.4f}.")
                else:
                    test_auc = temp_auc_check_final
            y_pred = best_model_for_type.predict(X_test_df)
        else: # Case where y_pred_proba_raw was None
            test_auc = 0.0 # Cannot compute AUC
            y_pred = best_model_for_type.predict(X_test_df) # Still get class predictions


        # Calculate other metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # --- Sanitize final metrics before returning ---
        final_metrics = {
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1)
        }
        # Use np.nan_to_num for these as well
        for metric_name, value in final_metrics.items():
            final_metrics[metric_name] = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
            if final_metrics[metric_name] < 0: # Ensure non-AUC metrics are not negative
                final_metrics[metric_name] = 0.0

        # Also sanitize test_auc
        test_auc = float(np.nan_to_num(test_auc, nan=0.0, posinf=0.0, neginf=0.0))


        return test_auc, best_model_for_type, study.best_params, final_metrics

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Main function to initiate model training across various algorithms.
        Manages MLflow runs for overall best model and individual model types.
        """
        try:
            logging.info("Splitting training and test arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Initialize with small positive float to ensure valid values from start
            best_overall_model_score = float(EPSILON)
            best_overall_model = None
            best_overall_model_name = None
            best_overall_model_params = {}
            best_overall_additional_metrics = { # Initialize with valid floats
                "test_accuracy": float(EPSILON),
                "test_precision": float(EPSILON),
                "test_recall": float(EPSILON),
                "test_f1": float(EPSILON)
            }

            model_names = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVM", "Logistic Regression"]

            # --- Conditional MLflow Parent Run Start ---
            # Only attempt to start the parent run if MLflow was successfully initialized.
            if MLFLOW_INITIALIZED:
                with mlflow.start_run(run_name="Best_Model_Run") as parent_run:
                    _run_mlflow_parent_logging = True
                    logging.info("MLflow parent run 'Best_Model_Run' started.")
            else:
                _run_mlflow_parent_logging = False
                logging.warning("MLflow not initialized. Skipping parent run logging.")

            # Loop through models whether MLflow is active or not
            for model_name in model_names:
                # If parent run isn't active, don't try nested runs, but still run optimization
                if MLFLOW_INITIALIZED and _run_mlflow_parent_logging:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        _run_mlflow_nested_logging = True
                else:
                    _run_mlflow_nested_logging = False

                try:
                    # Optimize and get test set performance of the best model for this type
                    test_auc, model, best_params, additional_metrics = self.optimize_model(
                        model_name, X_train, y_train, X_test, y_test
                    )

                    # Only log and consider for overall best if a valid model was returned AND test_auc is valid
                    # test_auc and additional_metrics are already sanitized by optimize_model
                    if model is not None and test_auc is not None and not np.isnan(test_auc) and not np.isinf(test_auc):
                        if _run_mlflow_nested_logging:
                            mlflow.log_params(best_params)
                            mlflow.log_metric("Best_AUC", float(test_auc)) # Explicit float cast
                            mlflow.log_metric("Best_Accuracy", float(additional_metrics["test_accuracy"]))
                            mlflow.log_metric("Best_Precision", float(additional_metrics["test_precision"]))
                            mlflow.log_metric("Best_Recall", float(additional_metrics["test_recall"]))
                            mlflow.log_metric("Best_F1-Score", float(additional_metrics["test_f1"]))

                            # Log the best model for this type as an artifact
                            model_path = os.path.join("temp_models", f"{model_name.replace(' ', '_')}_model.pkl")
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            save_object(file_path=model_path, obj=model)
                            mlflow.log_artifact(local_path=model_path, artifact_path="model_artifacts")
                            logging.info(f"Logged {model_name} model as artifact.")
                            os.remove(model_path) # Clean up temporary file

                        # Update overall best model if current model's AUC is better
                        if test_auc > best_overall_model_score: # test_auc is already sanitized here
                            best_overall_model = model
                            best_overall_model_score = test_auc
                            best_overall_model_name = model_name
                            best_overall_model_params = best_params
                            best_overall_additional_metrics = additional_metrics
                    else: # Optimization for this model failed completely or returned invalid AUC
                        logging.warning(f"Optimization for {model_name} returned no valid model or invalid AUC ({test_auc}). Skipping for overall best.")
                        if _run_mlflow_nested_logging:
                            mlflow.log_param(f"{model_name}_status", "Optimization Failed (No Valid Model/AUC)")
                            mlflow.log_metric(f"{model_name}_AUC", float(0.0)) # Log a placeholder score

                except Exception as model_optim_e:
                    logging.error(f"Overall optimization loop for model {model_name} failed: {model_optim_e}", exc_info=True)
                    if _run_mlflow_nested_logging:
                        mlflow.log_param(f"{model_name}_status", "Optimization Failed (Unhandled Exception)")
                        mlflow.log_metric(f"{model_name}_AUC", float(0.0)) # Log a placeholder score

            # After iterating through all models, log the overall best model to the parent run
            # This block will only execute if _run_mlflow_parent_logging is True.
            if _run_mlflow_parent_logging:
                if best_overall_model is None:
                    # This implies no model could be successfully optimized and kept.
                    logging.error("No overall best model could be determined as all model optimizations failed or returned invalid AUCs. MLflow parent run will be empty of final model data.")
                    # We can't raise ValueError here as it would break the MLflow run context.
                    # We just log a warning and ensure logged values are safe defaults.
                    mlflow.log_param("Overall_Best_Model", "N/A - No valid model found")
                    mlflow.log_metrics({
                        "Overall_Best_AUC": float(0.0),
                        "Overall_Best_Accuracy": float(0.0),
                        "Overall_Best_Precision": float(0.0),
                        "Overall_Best_Recall": float(0.0),
                        "Overall_Best_F1-Score": float(0.0)
                    })
                else:
                    save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_overall_model
                    )

                    mlflow.log_param("Overall_Best_Model", best_overall_model_name)

                    # --- Final check and sanitization for overall best metrics before logging ---
                    overall_metrics_to_log = {
                        "Overall_Best_AUC": float(best_overall_model_score),
                        "Overall_Best_Accuracy": float(best_overall_additional_metrics["test_accuracy"]),
                        "Overall_Best_Precision": float(best_overall_additional_metrics["test_precision"]),
                        "Overall_Best_Recall": float(best_overall_additional_metrics["test_recall"]),
                        "Overall_Best_F1-Score": float(best_overall_additional_metrics["test_f1"])
                    }

                    logging.info(f"DEBUG: Preparing to log overall_metrics_to_log: {overall_metrics_to_log}")

                    sanitized_overall_metrics = {}
                    for metric_name, value in overall_metrics_to_log.items():
                        float_value = float(value)
                        sanitized_value = float(np.nan_to_num(float_value, nan=0.0, posinf=0.0, neginf=0.0))
                        if sanitized_value < 0 and metric_name != "Overall_Best_AUC":
                            sanitized_value = 0.0
                        sanitized_overall_metrics[metric_name] = sanitized_value

                    logging.info(f"DEBUG: Logging sanitized_overall_metrics: {sanitized_overall_metrics}")

                    mlflow.log_metrics(sanitized_overall_metrics)
                    mlflow.log_artifact(local_path=self.model_trainer_config.trained_model_file_path, artifact_path="final_best_model")

            # This return value is consumed by data_ingestion.py
            logging.info(f"Best overall model: {best_overall_model_name} with AUC: {best_overall_model_score:.4f}")
            return best_overall_model_score

        except Exception as e:
            # Catch any top-level exceptions and re-raise as CustomException
            raise CustomException(e, sys)
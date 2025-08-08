import pytest
import numpy as np
from src.components.model_train import ModelTrainer

@pytest.fixture
def train_test_data():
    """
    Returns small dummy train/test datasets for fast CI testing.
    """
    # Dummy features (float)
    X_train = np.array([[0.1, 0.5], [0.9, 0.2], [0.4, 0.8], [0.7, 0.3]])
    X_test = np.array([[0.6, 0.4], [0.3, 0.9]])
    
    # Dummy labels (binary classification)
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([1, 0])
    
    # Concatenate to match expected train_arr/test_arr shape
    train_arr = np.hstack((X_train, y_train.reshape(-1, 1)))
    test_arr = np.hstack((X_test, y_test.reshape(-1, 1)))
    
    return train_arr, test_arr


def test_model_training(train_test_data):
    """
    Runs a minimal training test on dummy data with only Logistic Regression
    for speed in CI.
    """
    train_arr, test_arr = train_test_data

    # Disable MLflow/Dagshub logging in CI
    trainer = ModelTrainer(enable_logging=False)

    # Train only Logistic Regression for speed
    score = trainer.initiate_model_trainer(
        train_arr,
        test_arr,
        n_trials=1,
        experiment_name="ci_test_experiment"
    )

    assert score is not None, "Training did not return a score."
    assert score >= 0, f"Expected non-negative AUC score, got {score}"



#import os
#from src.components.model_train import ModelTrainer
#
#def test_model_training(train_test_data):
#    """
#    Test the model training pipeline.
#
#    - Uses enable_logging=False to skip Dagshub/MLflow logging during tests.
#    - Asserts that the returned AUC score is >= 0.
#    - You can toggle enable_logging=True for local testing to see metrics/tags in Dagshub.
#    """
#
#    # Load train and test arrays
#    train_arr, test_arr = train_test_data
#
#    # Toggle this flag to True when testing locally to see MLflow logs
#    enable_logging_local = False or bool(os.getenv("LOCAL_MLFLOW_LOGGING", False))
#
#    # Create trainer instance
#    trainer = ModelTrainer(enable_logging=enable_logging_local)
#
#    # Run training with minimal trials to speed up tests
#    score = trainer.initiate_model_trainer(train_arr, test_arr, n_trials=1)
#
#    # Basic check — ensure score is non-negative
#    assert score is not None, "Training did not return a score."
#    assert score >= 0, f"Expected non-negative AUC score, got {score}"
#
#    # Optional — if logging is enabled, print score for confirmation
#    if enable_logging_local:
#        print(f"✅ Test run completed with AUC score: {score}")
#
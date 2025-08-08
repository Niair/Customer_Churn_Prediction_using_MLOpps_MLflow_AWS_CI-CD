import os
from src.components.model_train import ModelTrainer

def test_model_training(train_test_data):
    """
    Test the model training pipeline.

    - Uses enable_logging=False to skip Dagshub/MLflow logging during tests.
    - Asserts that the returned AUC score is >= 0.
    - You can toggle enable_logging=True for local testing to see metrics/tags in Dagshub.
    """

    # Load train and test arrays
    train_arr, test_arr = train_test_data

    # Toggle this flag to True when testing locally to see MLflow logs
    enable_logging_local = False or bool(os.getenv("LOCAL_MLFLOW_LOGGING", False))

    # Create trainer instance
    trainer = ModelTrainer(enable_logging=enable_logging_local)

    # Run training with minimal trials to speed up tests
    score = trainer.initiate_model_trainer(train_arr, test_arr, n_trials=1)

    # Basic check — ensure score is non-negative
    assert score is not None, "Training did not return a score."
    assert score >= 0, f"Expected non-negative AUC score, got {score}"

    # Optional — if logging is enabled, print score for confirmation
    if enable_logging_local:
        print(f"✅ Test run completed with AUC score: {score}")

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.components.model_train import ModelTrainerConfig, ModelTrainer

@pytest.fixture
def sample_data():

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=2,
        n_redundant=10,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    train_arr = np.hstack((X[:800], y[:800].reshape(-1, 1)))
    test_arr = np.hstack((X[800:], y[800:].reshape(-1, 1)))
    return train_arr, test_arr

def test_model_trainer_init(sample_data):
    train_arr, test_arr = sample_data
    trainer = ModelTrainer()
    auc_score = trainer.initiate_model_trainer(train_arr, test_arr)
    
    assert isinstance(auc_score, float)
    assert 0.0 <= auc_score <= 1.0
    assert auc_score > 0.5

def test_optimization_flow(sample_data, mocker):
    train_arr, test_arr = sample_data
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
    
    trainer = ModelTrainer()
    test_auc, model, params, metrics = trainer.optimize_model(
        "Random Forest", X_train, y_train, X_test, y_test
    )
    
    assert isinstance(test_auc, float)
    assert test_auc > 0.5
    assert isinstance(params, dict)
    assert "test_accuracy" in metrics

def test_full_training_workflow(sample_data, tmp_path):
    train_arr, test_arr = sample_data
    trainer = ModelTrainer()
    trainer.model_trainer_config.trained_model_file_path = tmp_path / "model.pkl"
    
    auc_score = trainer.initiate_model_trainer(train_arr, test_arr)
    assert (tmp_path / "model.pkl").exists()
    assert auc_score > 0.6

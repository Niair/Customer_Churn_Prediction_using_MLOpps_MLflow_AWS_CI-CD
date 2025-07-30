import os
import numpy as np
from src.components.model_train import ModelTrainer

def test_model_trainer_runs_with_mock_data(tmp_path):
    np.random.seed(42)
    X = np.random.rand(30, 10)
    y = np.random.randint(0, 2, size=(30,))
    
    train_array = np.c_[X, y]
    test_array = np.c_[X, y]

    trainer = ModelTrainer()
    result = trainer.initiate_model_trainer(train_array, test_array)

    assert isinstance(result, dict)
    assert "best_model" in result
    assert "metrics" in result
    assert "model_path" in result
    assert os.path.exists(result["model_path"])

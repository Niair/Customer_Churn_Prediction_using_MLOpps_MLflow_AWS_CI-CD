import os
import numpy as np
from src.components.model_train import ModelTrainer

def test_model_trainer_runs_on_sample_data(tmp_path):
    # Create dummy train/test array: 5 samples, 3 features + 1 label
    X_dummy = np.random.rand(5, 3)
    y_dummy = np.array([0, 1, 0, 1, 0]).reshape(-1, 1)
    train_arr = np.hstack((X_dummy, y_dummy))
    test_arr = np.hstack((X_dummy, y_dummy))

    trainer = ModelTrainer()
    auc_score = trainer.initiate_model_trainer(train_arr, test_arr)

    assert isinstance(auc_score, float)
    assert 0.0 <= auc_score <= 1.0
    assert os.path.exists(trainer.model_trainer_config.trained_model_file_path)

# test_model_train.py
from src.components.model_train import ModelTrainer

def test_model_training(train_test_data):
    train_arr, test_arr = train_test_data
    trainer = ModelTrainer(enable_logging=False)  # disable Dagshub/MLflow logging
    score = trainer.initiate_model_trainer(train_arr, test_arr, n_trials=1)
    assert score > 0

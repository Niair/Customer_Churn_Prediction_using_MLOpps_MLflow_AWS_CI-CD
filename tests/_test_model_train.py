import pytest
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrainer

def test_model_training_pipeline():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_ingestion_config()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    result = trainer.initiate_model_trainer(train_arr, test_arr)

    assert "model" in result
    assert "metrics" in result
    assert "model_path" in result
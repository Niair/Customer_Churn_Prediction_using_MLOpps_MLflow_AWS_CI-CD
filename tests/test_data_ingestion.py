import os
import pytest
from src.components.data_ingestion import DataIngestion

def test_data_ingestion_outputs():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_ingestion_config()

    assert os.path.exists(train_path), "Train data file not created."
    assert os.path.exists(test_path), "Test data file not created."
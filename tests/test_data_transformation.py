import numpy as np
import pytest
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

def test_data_transformation_shapes():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_ingestion_config()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

    assert isinstance(train_arr, np.ndarray)
    assert isinstance(test_arr, np.ndarray)
    assert train_arr.shape[1] == test_arr.shape[1], "Train and test shapes must match."
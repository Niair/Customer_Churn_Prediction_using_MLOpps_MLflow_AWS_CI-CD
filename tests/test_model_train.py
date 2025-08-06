import pandas as pd
import numpy as np
import pytest
import os

from src.components.model_train import ModelTrainerConfig, ModelTrainer

# test_model_train.py
@pytest.fixture(scope="module")
def train_test_data():
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"
    
    assert os.path.exists(train_path), f"Missing file: {train_path}"
    assert os.path.exists(test_path), f"Missing file: {test_path}"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Validate training data
    y_train = train_df.iloc[:, -1]
    if y_train.nunique() < 2:
        pytest.fail("Training data needs at least 2 classes")
    if y_train.sum() < 2:
        pytest.fail("Training data needs at least 2 positive samples for SMOTE")

    # Validate test data
    y_test = test_df.iloc[:, -1]
    if y_test.nunique() < 2:
        pytest.fail("Test data needs at least 2 classes")

    return train_df.values, test_df.values
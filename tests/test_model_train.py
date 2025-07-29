import os
import pandas as pd
from src.components.model_trainer import ModelTrainer

def test_model_trainer_runs_on_sample_data(tmp_path):
    # Sample mock data with correct schema
    sample_data = pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'tenure': [1, 34, 2],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 53.85],
        'TotalCharges': ['29.85', '1889.5', '108.15'],
        'Churn': [0, 1, 0]
    })

    # Prepare X and y
    X = sample_data.drop('Churn', axis=1)
    y = sample_data['Churn']
    data_array = pd.concat([X, y], axis=1).values

    # Create a mock model trainer
    trainer = ModelTrainer()
    result = trainer.initiate_model_training(data_array, data_array)  # train_array = test_array for mock

    # Assertions
    assert isinstance(result, dict)
    assert "metrics" in result
    assert "model_path" in result
    assert os.path.exists(result["model_path"])

import pytest
import pandas as pd
import joblib
from pathlib import Path

# Toggle for using XGBoost instead of RandomForest
USE_XGBOOST = False

@pytest.fixture(scope="module")
def model_and_preprocessor():
    """Load the trained model and preprocessor used in production."""
    model_path = Path("artifacts/model.pkl")
    preprocessor_path = Path("artifacts/preprocessor.pkl")

    if not model_path.exists() or not preprocessor_path.exists():
        pytest.fail("Model or preprocessor not found. Run training before tests.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

@pytest.fixture(scope="module")
def test_data():
    """Load a small sample from test.csv."""
    test_csv_path = Path("artifacts/test.csv")
    if not test_csv_path.exists():
        pytest.fail("Test CSV not found in artifacts.")
    
    df = pd.read_csv(test_csv_path)
    return df.sample(5, random_state=42)  # Take 5 random rows

def test_predict_pipeline_runs(model_and_preprocessor, test_data):
    model, preprocessor = model_and_preprocessor

    # Separate features and target if target exists
    if "target" in test_data.columns:
        X = test_data.drop(columns=["target"])
    else:
        X = test_data

    # Apply preprocessing before prediction
    X_processed = preprocessor.transform(X)

    # Predict
    preds = model.predict(X_processed)

    assert preds is not None
    assert len(preds) == len(X)
    assert all(p in [0, 1] for p in preds)

    # Example XGBoost usage if needed:
    # if USE_XGBOOST:
    #     import xgboost as xgb
    #     xgb_model = joblib.load("artifacts/xgb_model.pkl")
    #     preds = xgb_model.predict(xgb.DMatrix(X_processed))

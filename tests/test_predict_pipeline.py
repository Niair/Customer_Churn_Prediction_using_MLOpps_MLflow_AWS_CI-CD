from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.components.model_train import ModelTrainerConfig, ModelTrainer

# -------------------- Prediction Pipeline Test --------------------

def test_predict_pipeline_runs():
    data = CustomData(
        monthly_charge=70.0,
        zip_code=90001,
        longitude=-118.25,
        age=35,
        latitude=34.05,
        total_long_distance_charges=30.0,
        tenure_in_months=24,
        total_revenue=500.0,
        number_of_referrals=2,
        total_charges=600.0,
        avg_monthly_long_distance_charges=10.0,
        avg_monthly_gb_download=20.0,
        number_of_dependents=1,
        engagement_score=5.0,
        num_addon_services=2,
        city="Los Angeles",
        contract="Month-to-Month",
        payment_method="Credit Card",
        offer="None",
        paperless_billing="Yes",
        gender="Male",
        married="Yes",
        internet_type="Fiber Optic"
    )

    df = data.get_data_as_data_frame()
    predictor = PredictPipeline()
    prediction = predictor.predict(df)

    assert prediction[0] in [0, 1]

# -------------------- Config Object Test --------------------

def test_model_trainer_config_paths():
    config = ModelTrainerConfig()
    assert config.trained_model_file_path.endswith(".pkl"), "Model path must end with .pkl"

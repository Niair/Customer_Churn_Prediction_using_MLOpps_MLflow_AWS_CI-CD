import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def test_predict_pipeline_runs():
    data = CustomData(
        monthly_charge=70.0,
        city="Los Angeles",
        longitude=-118.25,
        latitude=34.05,
        zip_code=90001,
        number_of_referrals=2,
        age=35,
        tenure_in_months=24,
        total_long_distance_charges=30.0,
        total_revenue=500.0,
        avg_monthly_long_distance_charges=10.0,
        total_charges=600.0,
        avg_monthly_gb_download=20.0,
        contract="Month-to-Month",
        number_of_dependents=1,
        payment_method="Credit Card",
        engagement_score=5.0,
        num_addon_services=2,
        offer="None",
        total_extra_data_charges=40.0,
        paperless_billing="Yes",
        total_refunds=10.0,
        multiple_lines="Yes"
    )

    df = data.get_data_as_data_frame()
    predictor = PredictPipeline()
    prediction = predictor.predict(df)

    assert prediction[0] in [0, 1]
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        monthly_charge: float,
        zip_code: int,
        longitude: float,
        age: int,
        latitude: float,
        total_long_distance_charges: float,
        tenure_in_months: int,
        total_revenue: float,
        number_of_referrals: int,
        total_charges: float,
        avg_monthly_long_distance_charges: float,
        avg_monthly_gb_download: float,
        number_of_dependents: int,
        engagement_score: float,
        num_addon_services: int,
        city: str,
        contract: str,
        payment_method: str,
        offer: str,
        paperless_billing: str,
        gender: str,
        married: str,
        internet_type: str,
    ):
        self.monthly_charge = monthly_charge
        self.zip_code = zip_code
        self.longitude = longitude
        self.age = age
        self.latitude = latitude
        self.total_long_distance_charges = total_long_distance_charges
        self.tenure_in_months = tenure_in_months
        self.total_revenue = total_revenue
        self.number_of_referrals = number_of_referrals
        self.total_charges = total_charges
        self.avg_monthly_long_distance_charges = avg_monthly_long_distance_charges
        self.avg_monthly_gb_download = avg_monthly_gb_download
        self.number_of_dependents = number_of_dependents
        self.engagement_score = engagement_score
        self.num_addon_services = num_addon_services
        self.city = city
        self.contract = contract
        self.payment_method = payment_method
        self.offer = offer
        self.paperless_billing = paperless_billing
        self.gender = gender
        self.married = married
        self.internet_type = internet_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "monthly_charge": [self.monthly_charge],
                "zip_code": [self.zip_code],
                "longitude": [self.longitude],
                "age": [self.age],
                "latitude": [self.latitude],
                "total_long_distance_charges": [self.total_long_distance_charges],
                "tenure_in_months": [self.tenure_in_months],
                "total_revenue": [self.total_revenue],
                "number_of_referrals": [self.number_of_referrals],
                "total_charges": [self.total_charges],
                "avg_monthly_long_distance_charges": [self.avg_monthly_long_distance_charges],
                "avg_monthly_gb_download": [self.avg_monthly_gb_download],
                "number_of_dependents": [self.number_of_dependents],
                "engagement_score": [self.engagement_score],
                "num_addon_services": [self.num_addon_services],
                "city": [self.city],
                "contract": [self.contract],
                "payment_method": [self.payment_method],
                "offer": [self.offer],
                "paperless_billing": [self.paperless_billing],
                "gender": [self.gender],
                "married": [self.married],
                "internet_type": [self.internet_type],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

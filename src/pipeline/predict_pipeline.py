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
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

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
        city: str,
        longitude: float,
        latitude: float,
        zip_code: int,
        number_of_referrals: int,
        age: int,
        tenure_in_months: int,
        total_long_distance_charges: float,
        total_revenue: float,
        avg_monthly_long_distance_charges: float,
        total_charges: float,
        avg_monthly_gb_download: float,
        contract: str,
        number_of_dependents: int,
        payment_method: str,
        engagement_score: float,
        num_addon_services: int,
        offer: str,
        total_extra_data_charges: float,
        paperless_billing: str,
        total_refunds: float,
        multiple_lines: str,
    ):
        self.monthly_charge = monthly_charge
        self.city = city
        self.longitude = longitude
        self.latitude = latitude
        self.zip_code = zip_code
        self.number_of_referrals = number_of_referrals
        self.age = age
        self.tenure_in_months = tenure_in_months
        self.total_long_distance_charges = total_long_distance_charges
        self.total_revenue = total_revenue
        self.avg_monthly_long_distance_charges = avg_monthly_long_distance_charges
        self.total_charges = total_charges
        self.avg_monthly_gb_download = avg_monthly_gb_download
        self.contract = contract
        self.number_of_dependents = number_of_dependents
        self.payment_method = payment_method
        self.engagement_score = engagement_score
        self.num_addon_services = num_addon_services
        self.offer = offer
        self.total_extra_data_charges = total_extra_data_charges
        self.paperless_billing = paperless_billing
        self.total_refunds = total_refunds
        self.multiple_lines = multiple_lines

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "monthly_charge": [self.monthly_charge],
                "city": [self.city],
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "zip_code": [self.zip_code],
                "number_of_referrals": [self.number_of_referrals],
                "age": [self.age],
                "tenure_in_months": [self.tenure_in_months],
                "total_long_distance_charges": [self.total_long_distance_charges],
                "total_revenue": [self.total_revenue],
                "avg_monthly_long_distance_charges": [self.avg_monthly_long_distance_charges],
                "total_charges": [self.total_charges],
                "avg_monthly_gb_download": [self.avg_monthly_gb_download],
                "contract": [self.contract],
                "number_of_dependents": [self.number_of_dependents],
                "payment_method": [self.payment_method],
                "engagement_score": [self.engagement_score],
                "num_addon_services": [self.num_addon_services],
                "offer": [self.offer],
                "total_extra_data_charges": [self.total_extra_data_charges],
                "paperless_billing": [self.paperless_billing],
                "total_refunds": [self.total_refunds],
                "multiple_lines": [self.multiple_lines],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

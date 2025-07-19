import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        Create preprocessing pipeline for given columns
        '''
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="drop"
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            
            # Define expected columns
            expected_columns = {
                'numerical': [
                    'age', 'number_of_dependents', 'zip_code', 'latitude', 'longitude',
                    'number_of_referrals', 'tenure_in_months', 'avg_monthly_long_distance_charges',
                    'avg_monthly_gb_download', 'monthly_charge', 'total_charges',
                    'total_refunds', 'total_extra_data_charges', 'total_long_distance_charges',
                    'total_revenue', 'has_offer', 'offer_popularity'
                ],
                'categorical': [
                    'city', 'contract', 'paperless_billing', 'tenure_category'
                ],
                'target': 'customer_status'
            }
            
            # Validate columns exist in data
            def validate_columns(df, expected_cols):
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns: {missing_cols}")
                return True
            
            # Validate all expected columns exist
            validate_columns(train_df, expected_columns['numerical'] + expected_columns['categorical'] + [expected_columns['target']])
            validate_columns(test_df, expected_columns['numerical'] + expected_columns['categorical'] + [expected_columns['target']])
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(
                expected_columns['numerical'],
                expected_columns['categorical']
            )

            target_column = expected_columns['target']
            
            # Separate input features and target feature
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
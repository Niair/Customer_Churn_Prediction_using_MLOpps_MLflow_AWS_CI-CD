import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
      preprocessor_obj_file_path = os.path.join("artifacts", 'preprocessor.pkl')

class DataTrnsformation:

      def __init__(self):
            self.data_transformation_config = DataTransformationConfig()
      
      def get_data_transformer_object(self):
            '''
            This function is used for data transformation

            '''
            try:
                  num_cols = ['age', 'zip_code', 'tenure_in_months', 'longitude', 'latitude', 'monthly_charge', 'total_charges', 'total_long_distance_charges', 'avg_monthly_gb_download', 'total_revenue', 'number_of_dependents', 'total_refunds', 'num_addon_services', 'avg_monthly_long_distance_charges', 'engagement_score']
                  cat_cols = ['customer_status', 'city', 'customer_id', 'paperless_billing', 'contract', 'tenure_category']

                  num_pipeline = Pipeline(
                        steps = [
                              ("imputer", SimpleImputer(strategy = "median")),
                              ("scaler", StandardScaler())
                        ]
                  )
                  
                  cat_pipeline = Pipeline(
                        steps = [
                              ("impute", SimpleImputer(strategy = "most_frequent")),
                              ("encoder", OneHotEncoder())
                        ]
                  )
                  logging.info("Numerical columns : {num_cols}")
                  logging.info("Categorical columns : {cat_cols}")

                  preprocessor = ColumnTransformer(
                        [
                              ("numerical_cols", num_pipeline, num_cols),
                              ('categorical_cols', cat_pipeline, cat_cols)
                        ]
                  )
                  return preprocessor


            except Exception as e:
                  raise CustomException(e, sys)
      
      def initiate_data_transformation(self, train_path, test_path):

            try:
                  train_df = pd.read_csv(train_path)
                  test_df = pd.read_csv(test_path)

                  logging.info("Reading train and test data (completed)")

                  logging.info("Obtaining the preprocessing object (ongoing)")

                  preprocessor_object = self.get_data_transformer_object()

                  target_column_name = "customer_status"
                  numerical_columns = ['age', 'number_of_dependents', 'zip_code', 'latitude', 'longitude', 'number_of_referrals', 'tenure_in_months', 'avg_monthly_long_distance_charges', 'avg_monthly_gb_download', 'monthly_charge', 'total_charges', 'total_refunds', 'total_extra_data_charges', 'total_long_distance_charges', 'total_revenue', 'has_offer', 'offer_popularity']

                  input_features_train_df = train_df.drop(columns = [target_column_name], axis = 1)
                  target_feature_train_df = train_df[target_column_name]

                  input_features_test_df = test_df.drop(columns = [target_column_name], axis = 1)
                  target_feature_test_df = test_df[target_column_name]

                  logging.info("Appling the preprocessing object on training and testing data frame (completed)")

                  input_feature_train_arr = preprocessor_object.fit_transform(input_features_train_df)
                  input_feature_test_arr = preprocessor_object.transform(input_features_test_df)

                  train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                  test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                  logging.info("Saved Preprocessing object (ongoing)")

                  save_object(
                        file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_object
                  )

                  return (
                        train_arr,
                        test_arr,
                        self.data_transformation_config.preprocessor_obj_file_path,
                  )


            except Exception as e:
                  raise CustomException(e, sys)
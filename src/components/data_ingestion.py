import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_train import ModelTrainerConfig, ModelTrainer


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
      train_data_path: str = os.path.join('artifacts', 'train.csv')
      test_data_path: str = os.path.join('artifacts', 'test.csv')
      raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
      
      def __init__(self):
            self.ingestion_config = DataIngestionConfig()
      
      def initiate_ingestion_config(self):
            logging.info("Enter in the data ingestion method or component")
            try:
                  df = pd.read_csv(os.path.join(os.getcwd(), "data", "processed", "best_features_customer_churn_data.csv"))
                  logging.info("Reading the dataset as data frame")

                  os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

                  df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

                  logging.info("train test split iniciated")
                  logging.info(f"Original columns: {df.columns.tolist()}")
                  train_set, test_set = train_test_split(df, test_size = 0.2, stratify=df['customer_status'], random_state = 42)

                  train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
                  test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

                  logging.info("The data ingestion is completed")

                  return(
                        self.ingestion_config.train_data_path,
                        self.ingestion_config.test_data_path      
                  )

            except Exception as e:
                  raise CustomException(e,sys)
            

if __name__ == '__main__':
      obj = DataIngestion()
      train_data, test_data = obj.initiate_ingestion_config()

      data_transformation = DataTransformation()
      train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

      modeltrainer = ModelTrainer()
      print(modeltrainer.initiate_model_trainer(train_arr, test_arr))


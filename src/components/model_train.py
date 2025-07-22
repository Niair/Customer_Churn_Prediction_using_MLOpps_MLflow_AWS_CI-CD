import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.svm import SVC

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, r2_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainingConfig:
      trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

      def __init__(self):
            self.model_trainer_config = ModelTrainingConfig()
      
      def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
            try:
                  logging.info("Splitting the train and test input data")
                  X_train, y_train, X_test, y_test = (
                        train_arr[:,:-1],
                        train_arr[:,-1],
                        test_arr[:,:-1],
                        test_arr[:,-1]
                  )

                  models = {
                        "Random Forest": RandomForestClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Logistic Regression": LogisticRegression(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "XGBoost": XGBClassifier(),
                        "CatBoost": CatBoostClassifier(),
                        "LightGBMClassifier": LGBMClassifier(),
                        "SCM": SVC()
                  }

                  model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models = models)

            except Exception as e:
                  raise CustomException(e, sys)

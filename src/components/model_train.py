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

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainingConfig:
      trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

      def __init__(self):
            self.model_trainer_config = ModelTrainingConfig()
      
      def initiate_model_trainer(self, train_arr, test_arr, ):
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
                        # "LightGBMClassifier": lgb.LGBMClassifier(),
                        "SCM": SVC(probability=True)
                  }

                  model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

                  best_model_score = max(sorted(model_report.values()))

                  best_model_name = list(model_report.keys())[
                        list(model_report.values()).index(best_model_score)
                  ]

                  best_model = models[best_model_name]

                  if best_model_score < 0.6:
                        raise CustomException("No best model found", sys)    
                  
                  logging.info(f"Best model on train and test data")

                  save_object (
                        file_path = self.model_trainer_config.trained_model_file_path,
                        obj = best_model
                  )


                  if hasattr(best_model, "predict_proba"):
                        predicted = best_model.predict_proba(X_test)

                        if predicted.shape[1] == 2:
                              auc_roc = roc_auc_score(y_test, predicted[:, 1])
                        else:
                              auc_roc = roc_auc_score(y_test, predicted, multi_class="ovr")
                  else:
                        y_score = best_model.decision_function(X_test)

                  if len(set(y_test)) == 2:
                        auc_roc = roc_auc_score(y_test, y_score)
                  else:
                        auc_roc = roc_auc_score(y_test, y_score, multi_class="ovr")


                  return auc_roc

            except Exception as e:
                  raise CustomException(e, sys)

import os
import sys
import dill
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException

def save_object(file_path, obj):
      try:
            dir_path = os.path.dirname(file_path)
            os .makedirs(dir_path, exist_ok = True)

            with open(file_path, "wb") as file_obj:
                  dill.dump(obj, file_obj)

      except Exception as e:
            raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            metrics = {
                "Train ROC AUC": roc_auc_score(y_train, y_train_pred),
                "Test ROC AUC": roc_auc_score(y_test, y_test_pred),
                "Train Accuracy": accuracy_score(y_train, y_train_pred),
                "Test Accuracy": accuracy_score(y_test, y_test_pred),
                "Test Precision": precision_score(y_test, y_test_pred, average='binary'),
                "Test Recall": recall_score(y_test, y_test_pred, average='binary'),
                "Test F1-Score": f1_score(y_test, y_test_pred, average='binary'),
                "Test Confusion Matrix": confusion_matrix(y_test, y_test_pred).tolist(),  # convert to list to avoid serialization issues
                "Test Classification Report": classification_report(y_test, y_test_pred, output_dict=True),
                "Model Params": model.get_params()
            }

            report[name] = metrics

        return report

    except Exception as e:
        raise CustomException(e, sys)
      
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
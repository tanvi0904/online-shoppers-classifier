"""Shared bits for training and the Streamlit app: column lists, target parsing, metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef,
)

TARGET = "Revenue"

# heavy right-skewed counts/durations -> log1p before scaling
SKEWED_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "PageValues",
]
RATE_COLS = ["BounceRates", "ExitRates", "SpecialDay"]
# integer codes in the raw data, but they are labels, not quantities
CATEGORICAL_COLS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]

FEATURES = SKEWED_COLS + RATE_COLS + CATEGORICAL_COLS

# model name -> the file it gets saved as in model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def to_binary(series: pd.Series) -> np.ndarray:
    """Revenue/Weekend come as True/False, 'TRUE'/'FALSE' or 1/0 depending on who saved the CSV."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).astype(int).to_numpy()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    # Weekend can be True/False text, so turn it into 0/1
    X["Weekend"] = to_binary(X["Weekend"])
    # as strings, so the one-hot encoder treats the codes as labels
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str)
    return X


def compute_metrics(y_true, y_pred, y_score) -> dict:
    # y_score = predicted probability of a purchase (AUC needs it)
    # zero_division=0 so a model that never predicts a purchase doesn't crash
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

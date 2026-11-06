"""Shared bits for training and the Streamlit app: column lists, target parsing, metrics."""
import numpy as np
import pandas as pd

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

def to_binary(series: pd.Series) -> np.ndarray:
    """Revenue/Weekend come as True/False, 'TRUE'/'FALSE' or 1/0 depending on who saved the CSV."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).astype(int).to_numpy()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["Weekend"] = to_binary(X["Weekend"])
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str)
    return X

"""Train the 5 classifiers on the UCI Online Shoppers Purchasing Intention dataset.

Run:  python train_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from shopper_utils import (
    CATEGORICAL_COLS, RATE_COLS, SKEWED_COLS, TARGET,
    clean_features,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
SEED = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"downloading dataset from {DATA_URL}")
        DATA_PATH.parent.mkdir(exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("skewed", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), SKEWED_COLS),
        ("rates", StandardScaler(), RATE_COLS),
        # dense output so GaussianNB can use it too
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])


def main():
    df = load_data().drop_duplicates()
    print(f"dataset: {df.shape[0]} rows x {df.shape[1] - 1} features")
    print("class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    X_train = clean_features(train_df)
    print("features after preprocessing:", build_preprocessor().fit_transform(X_train).shape[1])


if __name__ == "__main__":
    main()

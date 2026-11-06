"""Train the 5 classifiers on the UCI Online Shoppers Purchasing Intention dataset.

Run:  python train_models.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from shopper_utils import (
    CATEGORICAL_COLS, MODEL_FILES, RATE_COLS, SKEWED_COLS, TARGET,
    clean_features, compute_metrics, to_binary,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
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


def build_models(cv) -> dict:
    # only ~15% of sessions end in a purchase, so balance classes where the model supports it
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Decision Tree": GridSearchCV(
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 10, 30]},
            scoring="f1", cv=cv, n_jobs=-1,
        ),
        "KNN": GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors": [5, 11, 21, 31], "weights": ["uniform", "distance"]},
            scoring="f1", cv=cv, n_jobs=-1,
        ),
    }


def main():
    df = load_data().drop_duplicates()
    print(f"dataset: {df.shape[0]} rows x {df.shape[1] - 1} features")
    print("class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    X_train, y_train = clean_features(train_df), to_binary(train_df[TARGET])
    X_test, y_test = clean_features(test_df), to_binary(test_df[TARGET])

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, clf in build_models(cv).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        if isinstance(clf, GridSearchCV):
            print(f"{name} best params: {clf.best_params_}")
            # store the refit best estimator, not the whole search object
            pipe.steps[-1] = ("clf", clf.best_estimator_)

        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_score)
        print(name, {k: round(v, 4) for k, v in metrics.items()})
        joblib.dump(pipe, MODEL_DIR / MODEL_FILES[name])
        print(f"saved {name}")


if __name__ == "__main__":
    main()

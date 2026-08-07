"""
Phase 4 — preprocessing and scaling.

Order of operations matters more than the transforms themselves:

  1. Split FIRST, stratified on churn. Any statistic learned from the
     test set (a mean, a scale, a category list, a SMOTE neighbour) is
     leakage, and it inflates your reported score without improving the
     model a single customer will actually see.
  2. Fit the scaler and encoder on TRAIN ONLY, then transform both.
  3. Resample (SMOTE) inside the training fold only — never on test,
     never before the split. Synthetic minority rows in a test set mean
     you are grading yourself on customers who do not exist.

Which models need scaling
-------------------------
  Needs it     : Logistic Regression (gradient scale + L2 penalty is
                 applied per-coefficient), SVM (RBF kernel is a distance),
                 KNN (distance).
  Doesn't      : Random Forest, XGBoost, any tree ensemble — splits are
                 threshold comparisons, invariant to monotone rescaling.
We scale in the shared ColumnTransformer regardless, because it costs
nothing for trees and keeps one preprocessing path for every model.

Run:  python src/preprocess.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from config import (DB_PATH, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,  # noqa: E402
                    TEST_SIZE)
from features import (ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC,  # noqa: E402
                      engineer_features)


def load_from_db() -> pd.DataFrame:
    """Pull the analyst view out of SQLite — the CSV is never read here."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM v_customer_360", conn)


def build_preprocessor() -> ColumnTransformer:
    """StandardScaler on numerics, one-hot on nominals.

    handle_unknown='ignore' matters at serve time: if the retention team
    ever sends a payment method that did not exist during training, the
    API returns a slightly less informed score instead of a 500.
    """
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), ENGINEERED_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), ENGINEERED_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_splits():
    df = load_from_db()
    X = engineer_features(df)
    y = df["churn_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"train {X_train.shape}  churn={y_train.mean():.4f}")
    print(f"test  {X_test.shape}  churn={y_test.mean():.4f}")
    print(f"class ratio (neg:pos) = {(1 - y_train.mean()) / y_train.mean():.2f} : 1")
    return X_train, X_test, y_train, y_test


def main() -> None:
    X_train, X_test, y_train, y_test = make_splits()

    pre = build_preprocessor()
    pre.fit(X_train)                       # fit on train only
    feature_names = list(pre.get_feature_names_out())
    print(f"\n{len(ENGINEERED_NUMERIC) + len(ENGINEERED_CATEGORICAL)} engineered columns "
          f"-> {len(feature_names)} model matrix columns after encoding")

    # Persist raw-engineered splits (for notebooks / reproducibility) and
    # the fitted transformer pieces the checklist asks for individually.
    X_train.assign(churn_flag=y_train.values).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(churn_flag=y_test.values).to_csv(PROCESSED_DIR / "test.csv", index=False)

    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(pre.named_transformers_["num"].named_steps["scale"], MODELS_DIR / "scaler.pkl")
    joblib.dump(pre.named_transformers_["cat"], MODELS_DIR / "encoder.pkl")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"\nwrote {PROCESSED_DIR/'train.csv'}")
    print(f"wrote {PROCESSED_DIR/'test.csv'}")
    print("wrote models/preprocessor.pkl, scaler.pkl, encoder.pkl, feature_names.json")


if __name__ == "__main__":
    main()

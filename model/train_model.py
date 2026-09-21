"""
train_model.py
----------------
Trains a gradient-boosted classifier (XGBoost) on the fused EHR + wearable
feature set to predict: "Will this patient's glucose spike in the next 2 hours?"

This is the core "algorithmic model" required by the challenge: it ingests both
static (EHR) and dynamic (wearable) streams and outputs an adverse-event
prediction, which powers the doctor-facing dashboard.

Why XGBoost over a deep sequence model (e.g. LSTM) for this PoC:
- Rolling-window features already encode short-term temporal signal
- Trains in seconds on CPU -> fast iteration for a hackathon timeline
- Feature importances are directly explainable to a clinical audience
  (important for doctor trust in a digital twin system)
- Handles the natural class imbalance (rare spikes) well with scale_pos_weight
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, average_precision_score
)
import xgboost as xgb
import joblib

FEATURE_COLS_EXCLUDE = ["patient_id", "timestamp", "spike_in_2h"]


def load_data(path="fused_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def split_by_patient(df, test_size=0.2, seed=42):
    """Split by patient_id (not by row) to avoid leakage between train/test —
    a patient's data should never appear in both sets."""
    patients = df["patient_id"].unique()
    train_p, test_p = train_test_split(patients, test_size=test_size, random_state=seed)
    train_df = df[df["patient_id"].isin(train_p)]
    test_df = df[df["patient_id"].isin(test_p)]
    return train_df, test_df


def train():
    df = load_data()
    train_df, test_df = split_by_patient(df)

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]

    X_train, y_train = train_df[feature_cols], train_df["spike_in_2h"]
    X_test, y_test = test_df[feature_cols], test_df["spike_in_2h"]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )

    print("Training XGBoost model on fused EHR + wearable features...")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\n=== Evaluation on held-out patients ===")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Average Precision (PR-AUC): {average_precision_score(y_test, y_pred_proba):.4f}")

    # Feature importance -> explainability for clinicians
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n=== Top 10 most predictive features ===")
    print(importances.sort_values(ascending=False).head(10))

    joblib.dump(model, "spike_prediction_model.pkl")
    print("\nModel saved -> spike_prediction_model.pkl")

    return model, feature_cols


if __name__ == "__main__":
    train()

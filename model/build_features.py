"""
build_features.py
-------------------
This is the core "Digital Twin fusion" step: combines STATIC EHR data with
DYNAMIC wearable time-series data into a single feature table for the model.

For every timestamp in the wearable stream, we compute a rolling 2-hour window
of glucose/HR/HRV/sleep/steps (the patient's recent physiological trajectory),
then attach the patient's static EHR profile (age, BMI, HbA1c, meds, history).

The label (spike_in_2h) tells the model: "given what you know right now about
this patient (twin state), will they spike in the next 2 hours?"
"""

import pandas as pd
import numpy as np

WINDOW_SIZE = 24  # 24 samples * 5 min = 2 hours of lookback


def rolling_window_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling statistical features per patient over a 2h window."""
    g = group.sort_values("timestamp").reset_index(drop=True)

    feats = pd.DataFrame(index=g.index)
    for col in ["glucose_mg_dl", "heart_rate_bpm", "hrv_ms"]:
        roll = g[col].rolling(WINDOW_SIZE, min_periods=1)
        feats[f"{col}_mean_2h"] = roll.mean()
        feats[f"{col}_std_2h"] = roll.std().fillna(0)
        feats[f"{col}_slope_2h"] = g[col].diff(WINDOW_SIZE).fillna(0) / WINDOW_SIZE
        feats[f"{col}_last"] = g[col]

    feats["steps_sum_2h"] = g["step_count_5min"].rolling(WINDOW_SIZE, min_periods=1).sum()
    feats["sleep_stage_last"] = g["sleep_stage"]

    feats["patient_id"] = g["patient_id"]
    feats["timestamp"] = g["timestamp"]
    feats["spike_in_2h"] = g["spike_in_2h"]
    return feats


def build_fused_dataset(ehr_path: str, wearable_path: str) -> pd.DataFrame:
    ehr = pd.read_csv(ehr_path)
    wearable = pd.read_csv(wearable_path, parse_dates=["timestamp"])

    print("Computing rolling wearable window features per patient...")
    feature_frames = []
    for pid, group in wearable.groupby("patient_id"):
        feature_frames.append(rolling_window_features(group))
    dynamic_features = pd.concat(feature_frames, ignore_index=True)

    print("Fusing with static EHR features (the 'twin profile')...")
    ehr_encoded = ehr.copy()
    ehr_encoded["sex"] = (ehr_encoded["sex"] == "M").astype(int)

    fused = dynamic_features.merge(ehr_encoded, on="patient_id", how="left")

    return fused


if __name__ == "__main__":
    fused = build_fused_dataset("../data/synthetic_ehr.csv", "../data/synthetic_wearable.csv")
    fused.to_csv("fused_dataset.csv", index=False)
    print(f"Fused dataset shape: {fused.shape}")
    print(f"Positive rate: {fused['spike_in_2h'].mean():.4f}")

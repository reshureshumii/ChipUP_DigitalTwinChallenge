"""
generate_wearable.py
---------------------
Generates SYNTHETIC continuous wearable / CGM (Continuous Glucose Monitor) time-series
data, mimicking open-source formats similar to Apple Health / Google Fit / dedicated
CGM devices (e.g., Dexcom, Freestyle Libre export formats).

This is the dynamic/real-time data stream: glucose readings (every 5 min, as real CGMs
sample), heart rate variability (HRV), sleep stage, and step count.

Glucose "spike events" are deliberately embedded (triggered by simulated meals / stress)
so the prediction model has a real signal to learn — mimicking real post-prandial spikes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

SAMPLES_PER_DAY = 288  # every 5 minutes
DAYS_PER_PATIENT = 7


def simulate_patient_timeseries(patient_id: str, baseline_glucose: float, spike_risk: float):
    """
    Simulate one patient's week of wearable data.
    spike_risk: 0-1, probability scaling for how often meal-driven spikes occur
    (higher for patients with poor HbA1c control, tied back to EHR).
    """
    n = SAMPLES_PER_DAY * DAYS_PER_PATIENT
    start_time = datetime(2026, 1, 1)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n)]

    glucose = np.full(n, baseline_glucose, dtype=float)
    heart_rate = np.random.normal(72, 8, size=n).clip(45, 160)
    hrv_ms = np.random.normal(45, 12, size=n).clip(10, 120)
    steps = np.random.poisson(3, size=n)
    sleep_stage = np.zeros(n)  # 0 = awake, 1 = light, 2 = deep, 3 = REM

    spike_flags = np.zeros(n, dtype=int)  # ground truth: 1 if a spike STARTS 2h from here

    # Simulate meals roughly 3x/day -> possible glucose spikes ~30-60 min later
    for day in range(DAYS_PER_PATIENT):
        meal_offsets = [8 * 12, 13 * 12, 19 * 12]  # ~8am, 1pm, 7pm in 5-min steps
        for m_off in meal_offsets:
            idx = day * SAMPLES_PER_DAY + m_off
            if idx >= n:
                continue
            if np.random.rand() < spike_risk:
                spike_start = idx + np.random.randint(6, 12)  # 30-60 min post-meal
                spike_duration = np.random.randint(6, 18)     # 30-90 min duration
                spike_magnitude = np.random.uniform(50, 120)  # mg/dL rise

                end = min(spike_start + spike_duration, n)
                ramp = np.linspace(0, spike_magnitude, end - spike_start)
                glucose[spike_start:end] += ramp

                # Label: mark the point 2 hours (24 samples) BEFORE spike onset as
                # the "predict ahead" target -> this is what the model learns to flag
                label_idx = spike_start - 24
                if 0 <= label_idx < n:
                    spike_flags[label_idx] = 1

                # heart rate / HRV often shift slightly around big glucose swings
                heart_rate[spike_start:end] += np.random.normal(5, 2, size=end - spike_start)
                hrv_ms[spike_start:end] -= np.random.normal(5, 2, size=end - spike_start)

        # Simulate nightly sleep (11pm - 6am roughly)
        sleep_start = day * SAMPLES_PER_DAY + 23 * 12
        sleep_end = min(sleep_start + 7 * 12, n)
        if sleep_start < n:
            stages = np.random.choice([1, 2, 3], size=max(0, sleep_end - sleep_start),
                                       p=[0.5, 0.3, 0.2])
            sleep_stage[sleep_start:sleep_end] = stages
            steps[sleep_start:sleep_end] = 0
            heart_rate[sleep_start:sleep_end] -= 8  # resting HR drop during sleep

    glucose = glucose.clip(60, 400)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "glucose_mg_dl": np.round(glucose, 1),
        "heart_rate_bpm": np.round(heart_rate, 1),
        "hrv_ms": np.round(hrv_ms, 1),
        "step_count_5min": steps,
        "sleep_stage": sleep_stage.astype(int),
        "spike_in_2h": spike_flags,  # ground-truth label for supervised learning
    })
    return df


def generate_wearable_dataset(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """Generate wearable time-series for every patient in the EHR dataframe,
    scaling spike risk by HbA1c (worse control -> more/larger spikes)."""
    all_dfs = []
    for _, row in ehr_df.iterrows():
        # Map HbA1c (4.5-13) to a spike risk probability (0.15 - 0.75)
        risk = np.interp(row["hba1c_percent"], [4.5, 13.0], [0.15, 0.75])
        baseline = np.interp(row["hba1c_percent"], [4.5, 13.0], [95, 160])
        df = simulate_patient_timeseries(row["patient_id"], baseline, risk)
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    ehr = pd.read_csv("synthetic_ehr.csv")
    wearable_df = generate_wearable_dataset(ehr)
    wearable_df.to_csv("synthetic_wearable.csv", index=False)
    print(f"Generated {len(wearable_df)} wearable readings -> synthetic_wearable.csv")
    print(f"Positive spike labels: {wearable_df['spike_in_2h'].sum()}")

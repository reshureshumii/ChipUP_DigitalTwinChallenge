"""
generate_ehr.py
----------------
Generates SYNTHETIC (non-real) Electronic Health Record data for Type 2 Diabetes
patients, mimicking the structure of tools like Synthea. This is static/historical
data: demographics, past diagnoses, lab results, and genetic/family history markers.

No real patient data is used anywhere in this project (DPDP Act / HIPAA compliant
by design — everything here is randomly generated).
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_ehr_dataset(n_patients: int = 200) -> pd.DataFrame:
    """Generate a synthetic EHR dataset for n_patients."""

    patient_ids = [f"P{i:04d}" for i in range(1, n_patients + 1)]

    age = np.random.randint(30, 75, size=n_patients)
    sex = np.random.choice(["M", "F"], size=n_patients)
    bmi = np.round(np.random.normal(28, 5, size=n_patients).clip(16, 45), 1)

    # HbA1c: key diabetes marker (%) — higher means poorer long-term glucose control
    hba1c = np.round(np.random.normal(7.2, 1.3, size=n_patients).clip(4.5, 13.0), 1)

    family_history = np.random.choice([0, 1], size=n_patients, p=[0.4, 0.6])

    # Years since T2D diagnosis
    years_since_diagnosis = np.random.randint(0, 20, size=n_patients)

    on_insulin = np.random.choice([0, 1], size=n_patients, p=[0.65, 0.35])
    on_metformin = np.random.choice([0, 1], size=n_patients, p=[0.3, 0.7])

    hypertension = np.random.choice([0, 1], size=n_patients, p=[0.55, 0.45])
    cardiovascular_history = np.random.choice([0, 1], size=n_patients, p=[0.8, 0.2])

    # Fasting lipid panel
    ldl = np.round(np.random.normal(110, 30, size=n_patients).clip(50, 220), 1)
    hdl = np.round(np.random.normal(45, 12, size=n_patients).clip(20, 90), 1)

    creatinine = np.round(np.random.normal(0.9, 0.25, size=n_patients).clip(0.4, 2.5), 2)

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "hba1c_percent": hba1c,
        "family_history_diabetes": family_history,
        "years_since_t2d_diagnosis": years_since_diagnosis,
        "on_insulin": on_insulin,
        "on_metformin": on_metformin,
        "hypertension": hypertension,
        "cardiovascular_history": cardiovascular_history,
        "ldl_mg_dl": ldl,
        "hdl_mg_dl": hdl,
        "creatinine_mg_dl": creatinine,
    })

    return df


if __name__ == "__main__":
    df = generate_ehr_dataset(n_patients=200)
    df.to_csv("synthetic_ehr.csv", index=False)
    print(f"Generated {len(df)} synthetic EHR records -> synthetic_ehr.csv")
    print(df.head())

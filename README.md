# ChipUP — Digital Twin for Type 2 Diabetes Glucose Spike Prediction

**Digital Twin Challenge 2026** — Happiest Health Reimagining and Reforming Healthcare in India Summit

## Team Details

| Field | Details |
|---|---|
| Team Name | ChipUP |
| College / Incubator | Chennai Institute of Technology |
| Team Leader | Reshmi K |
| Team Members | Reshmi K, Rosiny C A, Varsha S, Yokhithaa S |
| Degree | B.E. Engineering Students (Electronics)|

## Project Title

**ChipUP Digital Twin — Predictive Glucose Spike Monitor for Type 2 Diabetes**

## Problem Statement & Healthcare Use Case

Type 2 Diabetes (T2D) affects over 100 million people in India, and one of its most
dangerous everyday risks is unpredictable post-prandial (after-meal) glucose spikes,
which over time damage the cardiovascular system, kidneys, and eyes. Most patients only
discover a spike *after* it has already happened, via a CGM reading or a lab test —
by then, the physiological damage is already underway and intervention is reactive.

Our Digital Twin creates a continuously updated virtual replica of a patient's
metabolic state by fusing:

- **Static/Historical data** (their long-term clinical profile): demographics, HbA1c,
  BMI, years since diagnosis, medication (insulin/metformin), comorbidities
  (hypertension, cardiovascular history), lipid panel, kidney function markers.
- **Dynamic/Real-time data** (their live physiological trajectory): continuous
  glucose readings, heart rate, heart rate variability (HRV), sleep stage, and step
  count, sampled every 5 minutes like a real CGM/wearable feed.

By fusing these two streams, the model predicts **whether a glucose spike will occur
within the next 2 hours**, giving patients and doctors a window to intervene
proactively (adjusting a meal, taking a short walk, or an insulin correction) rather
than reacting after the spike has already caused harm.

## Technical Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Data generation | NumPy, Pandas (synthetic EHR + CGM/wearable simulation) |
| Feature engineering | Rolling-window statistical features (Pandas) |
| Model | XGBoost (gradient-boosted decision trees) |
| Evaluation | scikit-learn (ROC-AUC, PR-AUC, classification report) |
| Dashboard mockup | HTML/CSS (conceptual doctor-facing UI) |

### Why XGBoost for this PoC
- Rolling-window features already capture short-term temporal signal, so a full deep
  sequence model (LSTM/Transformer) is not required to demonstrate the core concept
  within the challenge timeline.
- Trains in seconds on CPU, enabling fast iteration.
- Feature importances are directly interpretable — critical for clinician trust in
  a digital twin system, and something we've surfaced directly in the dashboard's
  "Digital Twin Reasoning" panel.
- Handles class imbalance well (real spikes are rare events) via `scale_pos_weight`.

## Data — Navigating the Sandbox Rules

No real patient data is used anywhere in this project, in compliance with the DPDP
Act and HIPAA. All data is synthetically generated:

- `data/generate_ehr.py` — generates synthetic EHR records (200 patients) mimicking
  the structure of tools like **Synthea**: demographics, HbA1c, medication, comorbidities.
- `data/generate_wearable.py` — generates synthetic CGM/wearable time-series (5-minute
  intervals, 7 days per patient) with embedded, realistic post-meal glucose spikes whose
  onset probability and magnitude scale with each patient's HbA1c (worse control →
  more frequent/larger spikes), mimicking Apple Health / Google Fit / dedicated CGM
  export formats.

## Project Structure

```
ChipUP_Chennai Institute of Technology/
├── README.md
├── LICENSE
├── data/
│   ├── generate_ehr.py          # synthetic static EHR generator
│   ├── generate_wearable.py     # synthetic dynamic wearable/CGM generator
│   ├── synthetic_ehr.csv        # generated output
│   └── synthetic_wearable.csv   # generated output
├── model/
│   ├── build_features.py        # fuses EHR + wearable into training features
│   ├── train_model.py           # trains & evaluates the XGBoost spike predictor
│   ├── fused_dataset.csv        # generated fused feature table
│   └── spike_prediction_model.pkl
├── dashboard/
│   └── dashboard_mockup.html    # conceptual doctor-facing UI
└── docs/
    ├── architecture_diagram.pdf  # (add before submission)
    ├── presentation.pdf          # (add before submission)
    └── demo_video_link.txt       # (add before submission)
```

## How to Run

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn xgboost joblib

# 2. Generate synthetic data
cd data
python3 generate_ehr.py
python3 generate_wearable.py

# 3. Build fused features and train the model
cd ../model
python3 build_features.py
python3 train_model.py

# 4. Open the dashboard mockup
cd ../dashboard
# open dashboard_mockup.html in any browser
```

## Model Performance (on held-out patients, synthetic data)

- ROC-AUC: ~0.77
- Feature importance highlights glucose trend slope, HRV decline, sleep quality,
  and long-term HbA1c control as the top predictive signals — aligning with known
  clinical intuition about glycemic variability.
- **Known limitation (by design, for transparency):** precision on the positive
  (spike) class is currently low at the default decision threshold, due to the
  natural rarity of spike events. Threshold tuning, cost-sensitive learning, and
  a larger synthetic cohort are identified next steps.

## Conceptual Dashboard

`dashboard/dashboard_mockup.html` demonstrates how a doctor would interact with a
patient's digital twin: live physiological signals, a glucose trend chart, a risk
badge with model confidence, and a plain-language "Digital Twin Reasoning" panel
that explains *why* the model is flagging risk — designed to build clinician trust
rather than presenting a black-box score.

## Open-Source License

This project is released under the MIT License — see [LICENSE](./LICENSE).

## Ethical & Privacy Note

All data used in this project is synthetically generated. No real patient records,
identifiable health information, or proprietary datasets were used at any stage of
development, in accordance with the challenge's sandbox rules and applicable data
protection regulations (DPDP Act, HIPAA).

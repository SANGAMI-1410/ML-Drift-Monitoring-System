import pandas as pd
import numpy as np
from scipy import stats
import json
import os

# ── 1. Load reference data ────────────────────────────────
reference_data = pd.read_csv("data/reference_data.csv")
print("✅ Reference data loaded:", reference_data.shape) 

# ── 2. Simulate NORMAL current data (no drift) ────────────
np.random.seed(42)
normal_current = pd.DataFrame({
    "sepal length (cm)": np.random.normal(5.8, 0.8, 100),
    "sepal width (cm)": np.random.normal(3.0, 0.4, 100),
    "petal length (cm)": np.random.normal(3.7, 1.7, 100),
    "petal width (cm)": np.random.normal(1.2, 0.6, 100)
})

# ── 3. Simulate DRIFTED current data ─────────────────────
drifted_current = pd.DataFrame({
    "sepal length (cm)": np.random.normal(7.5, 0.5, 100),
    "sepal width (cm)": np.random.normal(2.0, 0.3, 100),
    "petal length (cm)": np.random.normal(6.5, 0.8, 100),
    "petal width (cm)": np.random.normal(2.5, 0.3, 100)
})

# ── 4. Drift detection using KS Test ─────────────────────
def detect_drift(reference, current, threshold=0.05):
    results = {}
    drift_detected = False
    for col in reference.columns:
        stat, p_value = stats.ks_2samp(reference[col], current[col])
        drifted = p_value < threshold
        if drifted:
            drift_detected = True
        results[col] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "drift_detected": bool(drifted)
        }
    return drift_detected, results

os.makedirs("data/reports", exist_ok=True)

# ── 5. Check NORMAL data ──────────────────────────────────
print("\n📊 Running drift check on NORMAL data...")
normal_drift, normal_results = detect_drift(reference_data, normal_current)
with open("data/reports/normal_drift_report.json", "w") as f:
    json.dump({"drift_detected": bool(normal_drift), "features": normal_results}, f, indent=2)
print(f"✅ Normal drift report saved")
print(f"   Drift detected: {normal_drift}")

# ── 6. Check DRIFTED data ────────────────────────────────
print("\n📊 Running drift check on DRIFTED data...")
drifted_drift, drifted_results = detect_drift(reference_data, drifted_current)
with open("data/reports/drifted_drift_report.json", "w") as f:
    json.dump({"drift_detected": drifted_drift, "features": drifted_results}, f, indent=2)
print(f"✅ Drifted drift report saved")
print(f"   Drift detected: {drifted_drift}") 

# ── 7. Summary ───────────────────────────────────────────
print("\n" + "="*50)
print("DRIFT DETECTION SUMMARY")
print("="*50)
print(f"Normal data drift detected  : {normal_drift}")
print(f"Drifted data drift detected : {drifted_drift}")

if drifted_drift:
    print("\n🚨 ALERT: Data drift detected in production!")
    print("   Model may need retraining.")
else:
    print("\n✅ No drift detected. Model is healthy.")
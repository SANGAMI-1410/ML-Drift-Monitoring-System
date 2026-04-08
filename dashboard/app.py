import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy import stats

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="ML Monitor Dashboard",
    page_icon="🤖", 
    layout="wide"
)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Load reference data ───────────────────────────────────
@st.cache_data
def load_reference():
    return pd.read_csv("data/reference_data.csv")

reference_data = load_reference()

# ── Drift detection function ──────────────────────────────
def detect_drift(reference, current, threshold=0.05):
    results = {}
    drift_detected = False
    for col in reference.columns:
        stat, p_value = stats.ks_2samp(reference[col], current[col])
        drifted = bool(p_value < threshold)
        if drifted:
            drift_detected = True
        results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": drifted
        }
    return bool(drift_detected), results

# ── Title ─────────────────────────────────────────────────
st.title("🤖 ML Monitor Dashboard")
st.markdown("**Model:** Random Forest Classifier | **Dataset:** Iris | **Version:** v1")
st.divider()

# ── Row 1: Model Health Metrics ──────────────────────────
st.subheader("📊 Model Health")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "100%", "Healthy")
col2.metric("F1 Score", "1.0", "Healthy")
col3.metric("Model Version", "v1", "Active")
col4.metric("Status", "🟢 Online", "Running")

st.divider()

# ── Row 2: Live Prediction ────────────────────────────────
st.subheader("🔮 Live Prediction")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Enter flower measurements:**")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    if st.button("🔮 Predict", type="primary"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        class_names = ["Setosa", "Versicolor", "Virginica"]

        with col2:
            st.markdown("**Prediction Result:**")
            st.success(f"🌸 Predicted Class: **{class_names[prediction]}**")
            st.metric("Confidence", f"{max(probability)*100:.1f}%")
            st.markdown("**Class Probabilities:**")
            prob_df = pd.DataFrame({
                "Class": class_names,
                "Probability": [round(float(p), 4) for p in probability]
            })
            st.bar_chart(prob_df.set_index("Class"))

st.divider()

# ── Row 3: Drift Detection ────────────────────────────────
st.subheader("🚨 Drift Detection")
st.markdown("Simulate incoming production data and check for drift:")

drift_type = st.radio(
    "Select data type:",
    ["Normal Data (no drift expected)", "Drifted Data (drift expected)"],
    horizontal=True
)

if st.button("🔍 Run Drift Check", type="primary"):
    np.random.seed(42)

    if drift_type == "Normal Data (no drift expected)":
        current = pd.DataFrame({
            "sepal length (cm)": np.random.normal(5.8, 0.8, 100),
            "sepal width (cm)": np.random.normal(3.0, 0.4, 100),
            "petal length (cm)": np.random.normal(3.7, 1.7, 100),
            "petal width (cm)": np.random.normal(1.2, 0.6, 100)
        })
    else:
        current = pd.DataFrame({
            "sepal length (cm)": np.random.normal(7.5, 0.5, 100),
            "sepal width (cm)": np.random.normal(2.0, 0.3, 100),
            "petal length (cm)": np.random.normal(6.5, 0.8, 100),
            "petal width (cm)": np.random.normal(2.5, 0.3, 100)
        })

    drift_detected, results = detect_drift(reference_data, current)

    if drift_detected:
        st.error("🚨 ALERT: Data drift detected! Model may need retraining.")
    else:
        st.success("✅ No drift detected. Model is healthy.")

    st.markdown("**Feature-level Drift Results:**")
    results_df = pd.DataFrame([
        {
            "Feature": col,
            "KS Statistic": v["ks_statistic"],
            "P-Value": v["p_value"],
            "Drift": "🚨 Yes" if v["drift_detected"] else "✅ No"
        }
        for col, v in results.items()
    ])
    st.dataframe(results_df, use_container_width=True)

st.divider()
st.caption("ML Monitor Dashboard | Built with Streamlit | Drift detection via KS Test") 
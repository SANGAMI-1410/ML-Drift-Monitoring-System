from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np 
import os

# ── Load model and scaler ──────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "../model/scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ── Class names ────────────────────────────────────────────
class_names = ["setosa", "versicolor", "virginica"]

# ── Input schema ──────────────────────────────────────────
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="ML Monitor API",
    description="Iris classification model with monitoring",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "ML Monitor API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "random-forest-v1"}

@app.post("/predict")
def predict(data: IrisInput):
    # Prepare input
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return {
        "prediction": int(prediction),
        "class_name": class_names[prediction],
        "confidence": round(float(max(probability)), 4),
        "probabilities": {
            "setosa": round(float(probability[0]), 4),
            "versicolor": round(float(probability[1]), 4),
            "virginica": round(float(probability[2]), 4)
        }
    }
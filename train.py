import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler 
import pickle
import os

# ── 1. Load data ──────────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Save reference data for drift detection later (Day 4)
X.to_csv("data/reference_data.csv", index=False)
print("✅ Reference data saved")

# ── 2. Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. MLflow experiment ───────────────────────────────────
mlflow.set_experiment("ml-monitor-experiment")

with mlflow.start_run(run_name="random-forest-v1"):

    n_estimators = 100
    max_depth = 5
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset", "iris")

    # Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Save locally
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Model trained")
    print(f"   Accuracy  : {accuracy:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"✅ Model saved to model/")
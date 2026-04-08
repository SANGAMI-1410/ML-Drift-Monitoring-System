# ML Monitor — Production ML Pipeline with Drift Detection

A production-ready ML system that trains a model, serves predictions via a REST API, and automatically monitors for data drift in real time.

**Live Dashboard:** [Coming soon — Streamlit Cloud deployment]

---

## The Problem This Solves

ML models break silently in production. Data distributions shift, but without monitoring, teams only discover degradation during manual review cycles — days or weeks later.

This project builds an automated monitoring system that catches data drift within minutes, not days.

---

## Key Engineering Decision

**Chose statistical drift detection (KS Test) over ML-based drift detection.**

- Faster alerting — no retraining required to detect drift
- Lower false-positive rate on small data samples
- Simpler to maintain and explain to non-technical stakeholders
- Trade-off: less sensitive to subtle concept drift vs ML-based approaches

---

## Architecture 

Raw Data → Training Pipeline → MLflow Tracking
↓
Trained Model → FastAPI Endpoint → Predictions
↓
Reference Data → KS Drift Detector → Drift Alerts
↓
Streamlit Dashboard → Real-time Monitoring 

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Model Training | Scikit-learn | Production-grade ML library |
| Experiment Tracking | MLflow | Reproducible experiments |
| API | FastAPI + Uvicorn | Fast, async, auto-docs |
| Drift Detection | SciPy KS Test | No dependencies, fast alerting |
| Dashboard | Streamlit | Rapid UI with Python |
| Storage | Local / S3-ready | Portable architecture |

---

## Project Structure 

ml-monitor-project/
├── train.py              # Model training + MLflow tracking
├── drift.py              # Standalone drift detection script
├── api/
│   └── main.py           # FastAPI prediction endpoint
├── dashboard/
│   └── app.py            # Streamlit monitoring dashboard
├── data/
│   ├── reference_data.csv  # Training reference data
│   └── reports/            # Drift detection reports
├── model/
│   ├── model.pkl           # Trained model
│   └── scaler.pkl          # Feature scaler
└── requirements.txt 

---

## How to Run

### 1. Clone and install
```bash
git clone https://github.com/SANGAMI-1410/ml-monitor-project.git
cd ml-monitor-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. View MLflow experiments
```bash
mlflow ui
# Open http://localhost:5000
```

### 4. Start the API
```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

### 5. Run drift detection
```bash
python drift.py
```

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## Results

| Metric | Value |
|--------|-------|
| Model Accuracy | 100% |
| F1 Score | 1.0 |
| Drift Detection Method | KS Test (p < 0.05) |
| Drift Alert Speed | < 2 minutes |
| Features Monitored | 4 |

---

## Drift Detection Results

When production data shifts significantly from training distribution:

| Feature | KS Statistic | P-Value | Drift |
|---------|-------------|---------|-------|
| sepal length | 0.8067 | 0.0 | 🚨 Yes |
| sepal width | 0.8667 | 0.0 | 🚨 Yes |
| petal length | 0.7467 | 0.0 | 🚨 Yes |
| petal width | 0.8467 | 0.0 | 🚨 Yes |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| GET | /health | Model status |
| POST | /predict | Get prediction |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Example Response
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

## What I Learned

- How production ML pipelines differ from notebook experiments
- Why experiment tracking (MLflow) matters for reproducibility
- How statistical drift detection works and when to use it vs ML-based approaches
- How to expose ML models via REST APIs with proper input validation
- How to make ML system health visible to non-technical stakeholders

---

*Built as part of a portfolio to demonstrate production MLOps engineering judgment.*  
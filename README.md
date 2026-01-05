# MLOps Churn API (FastAPI + Scikit-learn)

Production-style ML inference API for telecom churn prediction.

## Run locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
2) Train model
python src/train.py
3) Start API
uvicorn src.app:app --reload --port 8001
Health check
curl -s http://127.0.0.1:8001/health
Expected:
{"status":"ok","model_loaded":true}
Predict
High-risk example:
curl -s -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 5,
    "monthly_charges": 85.2,
    "total_charges": 420.5,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": 1,
    "support_tickets_90d": 1
  }'
Low-risk example:
curl -s -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 48,
    "monthly_charges": 45.0,
    "total_charges": 2100.0,
    "contract_type": "Two year",
    "internet_service": "DSL",
    "payment_method": "Bank transfer (automatic)",
    "paperless_billing": 0,
    "support_tickets_90d": 0
  }'
Example outputs
{"churn_probability":0.6485,"churn_prediction":1,"threshold":0.5}
{"churn_probability":0.1663,"churn_prediction":0,"threshold":0.5}
Repo structure
src/train.py — trains and saves model
src/app.py — FastAPI inference service
Author
Rithika

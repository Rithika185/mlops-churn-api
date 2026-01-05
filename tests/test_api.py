from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_schema_and_types():
    payload = {
        "tenure_months": 48,
        "monthly_charges": 45.0,
        "total_charges": 2100.0,
        "contract_type": "Two year",
        "internet_service": "DSL",
        "payment_method": "Bank transfer (automatic)",
        "paperless_billing": 0,
        "support_tickets_90d": 0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # Required output keys
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "threshold" in data

    # Type/Range checks
    assert isinstance(data["churn_probability"], float)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0, 1)
    assert isinstance(data["threshold"], float)


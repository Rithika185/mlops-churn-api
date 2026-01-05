from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ----------------------------
# Paths + model loading
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = None  # will be loaded on startup


def load_model():
    """Load the trained sklearn pipeline/model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# ----------------------------
# API schema
# ----------------------------
class PredictRequest(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)

    contract_type: str
    internet_service: str
    payment_method: str

    paperless_billing: int = Field(..., ge=0, le=1)   # must be 0/1
    support_tickets_90d: int = Field(..., ge=0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="MLOps Churn API", version="1.0.0")


@app.on_event("startup")
def _startup():
    global model
    model = load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model
    if model is None:
        # This should not happen if startup ran, but keep it safe.
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {e}")

    # Convert request -> 1-row dataframe (this is the key to avoid 2D array errors)
    row = req.model_dump()
    X = pd.DataFrame([row])

    try:
        proba = float(model.predict_proba(X)[0][1])  # probability of churn class
    except Exception as e:
        # Return the exception so you can debug quickly
        raise HTTPException(status_code=500, detail=str(e))

    threshold = 0.5
    pred = int(proba >= threshold)

    return PredictResponse(
        churn_probability=proba,
        churn_prediction=pred,
        threshold=threshold,
    )


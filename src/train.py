import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ARTIFACT_DIR = Path("models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def make_synthetic_churn(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_months": rng.integers(0, 72, size=n),
        "monthly_charges": rng.normal(70, 30, size=n).clip(10, 200),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], size=n, p=[0.6, 0.25, 0.15]),
        "paperless_billing": rng.choice([0, 1], size=n, p=[0.35, 0.65]),
        "support_tickets_90d": rng.poisson(1.2, size=n).clip(0, 10),
    })

    # Build churn probability with realistic signals
    logits = (
        -0.03 * df["tenure_months"]
        + 0.015 * df["monthly_charges"]
        + 0.55 * (df["contract_type"] == "month-to-month").astype(int)
        + 0.25 * df["paperless_billing"]
        + 0.18 * df["support_tickets_90d"]
        - 1.2
    )
    p = 1 / (1 + np.exp(-logits))
    df["churn"] = (rng.random(n) < p).astype(int)
    return df

def train() -> None:
    df = make_synthetic_churn()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    num_cols = ["tenure_months", "monthly_charges", "paperless_billing", "support_tickets_90d"]
    cat_cols = ["contract_type"]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=200)

    pipe = Pipeline(steps=[("preproc", preproc), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    joblib.dump(pipe, ARTIFACT_DIR / "model.joblib")

    metrics = {"roc_auc": float(auc), "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    (ARTIFACT_DIR / "metrics.json").write_text(pd.Series(metrics).to_json())

    print(f"Saved model to {ARTIFACT_DIR/'model.joblib'} | ROC-AUC={auc:.3f}")

if __name__ == "__main__":
    train()


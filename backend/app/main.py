"""
NexaTel Churn Intelligence API (Phase 7).

Run locally:
    uvicorn app.main:app --reload --port 8000     (from backend/)
Docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .schemas import BatchIn, BatchOut, CustomerIn, PredictionOut
from .service import (get_eda_stats, get_model_comparison, get_service,
                      get_shap_importance)

app = FastAPI(
    title="NexaTel Churn Intelligence API",
    description=(
        "Scores a subscriber's probability of cancelling, explains the "
        "drivers behind that score, and recommends a retention action. "
        "Built on 7,043 historical subscriber records."
    ),
    version="1.0.0",
)

# In production set ALLOWED_ORIGINS to the deployed frontend URL.
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Three real profiles from the historical data, for one-click demos.
SAMPLE_CUSTOMERS = [
    {
        "label": "High risk — new fiber customer, month-to-month",
        "customer": {
            "gender": "Female", "senior_citizen": 0, "partner": "No",
            "dependents": "No", "tenure": 2, "phone_service": "Yes",
            "multiple_lines": "No", "internet_service": "Fiber optic",
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "Yes", "streaming_movies": "Yes",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 94.4, "total_charges": 188.8,
        },
    },
    {
        "label": "Medium risk — mid-tenure DSL, manual payment",
        "customer": {
            "gender": "Male", "senior_citizen": 0, "partner": "Yes",
            "dependents": "No", "tenure": 28, "phone_service": "Yes",
            "multiple_lines": "Yes", "internet_service": "DSL",
            "online_security": "No", "online_backup": "Yes",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Mailed check",
            "monthly_charges": 61.2, "total_charges": 1713.6,
        },
    },
    {
        "label": "Low risk — long-tenured, two-year contract, autopay",
        "customer": {
            "gender": "Male", "senior_citizen": 0, "partner": "Yes",
            "dependents": "Yes", "tenure": 62, "phone_service": "Yes",
            "multiple_lines": "Yes", "internet_service": "DSL",
            "online_security": "Yes", "online_backup": "Yes",
            "device_protection": "Yes", "tech_support": "Yes",
            "streaming_tv": "Yes", "streaming_movies": "Yes",
            "contract": "Two year", "paperless_billing": "No",
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 89.9, "total_charges": 5573.8,
        },
    },
]


@app.get("/", tags=["meta"])
def root():
    """A stranger opening the deployed link should land on the tool, not
    on a JSON blob. The service descriptor stays available at /api."""
    if (Path(__file__).resolve().parents[2] / "frontend" / "index.html").exists():
        return RedirectResponse("/app/")
    return service_info()


@app.get("/api", tags=["meta"])
def service_info():
    return {
        "service": "NexaTel Churn Intelligence API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/health", "/api/model-info", "/api/predict",
                      "/api/predict/batch", "/api/stats", "/api/samples"],
    }


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe. Render pings this to wake the free-tier dyno."""
    svc = get_service()
    return {"status": "healthy", "model": svc.model_name,
            "threshold": svc.threshold,
            "explanations": svc.explanations_enabled}


@app.get("/api/model-info", tags=["meta"])
def model_info():
    return get_service().model_info()


@app.post("/api/predict", response_model=PredictionOut, tags=["scoring"])
def predict(customer: CustomerIn):
    """Score a single subscriber and explain the score."""
    try:
        return get_service().predict(customer.model_dump())
    except Exception as exc:                                  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc


@app.post("/api/predict/batch", response_model=BatchOut, tags=["scoring"])
def predict_batch(payload: BatchIn):
    """Score a list of subscribers — the nightly-run entry point."""
    if not payload.customers:
        raise HTTPException(status_code=400, detail="No customers supplied.")
    if len(payload.customers) > 500:
        raise HTTPException(status_code=413, detail="Batch limit is 500 customers.")
    return get_service().predict_batch([c.model_dump() for c in payload.customers])


@app.get("/api/stats", tags=["dashboard"])
def stats():
    """EDA headline figures + SHAP importance, for the dashboard tab."""
    return {
        "eda": get_eda_stats(),
        "shap_importance": get_shap_importance(),
        "model_comparison": get_model_comparison(),
    }


@app.get("/api/samples", tags=["dashboard"])
def samples():
    return {"samples": SAMPLE_CUSTOMERS}


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@app.get("/api/figures/{name}", tags=["dashboard"])
def figure(name: str):
    """Serve an EDA/SHAP figure by filename (whitelisted to reports/figures)."""
    root = Path(__file__).resolve().parents[2] / "reports" / "figures"
    path = (root / name).resolve()
    if not str(path).startswith(str(root)) or not path.exists():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------
# The agent tool is mounted on the same service as the API. One Render
# deploy serves both, which removes the CORS hop entirely and means the
# browser never needs to know a second hostname. Mounted last so every
# /api/* route above keeps priority over the static catch-all.
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/ui", include_in_schema=False)
def ui():
    """Convenience redirect: /ui -> the agent tool."""
    return RedirectResponse("/app/")

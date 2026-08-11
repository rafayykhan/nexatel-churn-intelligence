"""
API tests.

Uses FastAPI's TestClient, so the real model artifacts are loaded — these
are the checks that would have caught a stale pickle or a threshold that
drifted away from the risk bands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402

client = TestClient(app)

HIGH_RISK = {
    "tenure": 2, "monthly_charges": 94.4, "total_charges": 188.8,
    "contract": "Month-to-month", "internet_service": "Fiber optic",
    "tech_support": "No", "payment_method": "Electronic check",
}
LOW_RISK = {
    "tenure": 62, "monthly_charges": 89.9, "total_charges": 5573.8,
    "contract": "Two year", "internet_service": "DSL", "tech_support": "Yes",
    "online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes",
    "payment_method": "Credit card (automatic)", "partner": "Yes", "dependents": "Yes",
}


def test_health_reports_the_loaded_model():
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model"]
    assert 0 < body["threshold"] < 1


def test_high_and_low_risk_profiles_separate_clearly():
    high = client.post("/api/predict", json=HIGH_RISK).json()
    low = client.post("/api/predict", json=LOW_RISK).json()
    assert high["churn_probability"] > low["churn_probability"]
    assert high["risk_level"] == "High"
    assert low["risk_level"] == "Low"
    assert high["flagged_for_outreach"] is True
    assert low["flagged_for_outreach"] is False


def test_risk_level_agrees_with_the_decision_threshold():
    body = client.post("/api/predict", json=HIGH_RISK).json()
    above = body["churn_probability"] >= body["decision_threshold"]
    assert above == (body["risk_level"] == "High") == body["flagged_for_outreach"]


def test_every_prediction_carries_a_reason_and_an_action():
    body = client.post("/api/predict", json=HIGH_RISK).json()
    assert body["risk_factors"], "a flagged customer with no reason is unusable to an agent"
    assert body["recommended_action"]
    for factor in body["risk_factors"]:
        assert factor["impact"] > 0
        assert factor["label"] and "_" not in factor["label"]


def test_reasons_are_not_repeated():
    """Contract appears as an ordinal and as a dummy; the agent sees it once."""
    body = client.post("/api/predict", json=HIGH_RISK).json()
    labels = [f["label"] for f in body["risk_factors"] + body["protective_factors"]]
    assert len(labels) == len(set(labels))


def test_total_charges_is_optional():
    payload = {k: v for k, v in HIGH_RISK.items() if k != "total_charges"}
    assert client.post("/api/predict", json=payload).status_code == 200


def test_brand_new_customer_scores_without_error():
    body = client.post("/api/predict", json={"tenure": 0, "monthly_charges": 45.3}).json()
    assert 0.0 <= body["churn_probability"] <= 1.0


@pytest.mark.parametrize("bad", [
    {"tenure": -5, "monthly_charges": 50},
    {"tenure": 5, "monthly_charges": -20},
    {"tenure": 5, "monthly_charges": 50, "contract": "Three year"},
    {"monthly_charges": 50},
])
def test_invalid_input_is_rejected_with_422(bad):
    assert client.post("/api/predict", json=bad).status_code == 422


def test_batch_summary_adds_up():
    body = client.post("/api/predict/batch",
                       json={"customers": [HIGH_RISK, LOW_RISK]}).json()
    s = body["summary"]
    assert s["scored"] == 2
    assert s["high_risk"] + s["medium_risk"] + s["low_risk"] == 2
    assert s["flagged_for_outreach"] == sum(
        r["flagged_for_outreach"] for r in body["results"])


def test_empty_batch_is_rejected():
    assert client.post("/api/predict/batch", json={"customers": []}).status_code == 400


def test_stats_endpoint_feeds_the_dashboard():
    body = client.get("/api/stats").json()
    assert body["eda"]["overall"]["customers"] == 7043
    assert body["shap_importance"]
    assert body["model_comparison"]


def test_figure_route_blocks_path_traversal():
    assert client.get("/api/figures/../../models/model.pkl").status_code == 404

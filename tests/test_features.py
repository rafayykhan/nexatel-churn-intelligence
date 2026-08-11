"""
Feature-layer tests.

These guard the things that silently break a churn model in production:
train/serve skew, leakage, and the tenure=0 division that only shows up
on a brand-new customer — exactly the customer retention cares about most.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DB_PATH  # noqa: E402
from features import MODEL_INPUT_COLUMNS, engineer_features  # noqa: E402

BASE = {
    "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
    "tenure": 12, "phone_service": "Yes", "multiple_lines": "No",
    "internet_service": "Fiber optic", "online_security": "No",
    "online_backup": "No", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "No", "streaming_movies": "No", "contract": "Month-to-month",
    "paperless_billing": "Yes", "payment_method": "Electronic check",
    "monthly_charges": 80.0, "total_charges": 960.0,
}


def frame(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **overrides}])


def test_output_schema_is_stable():
    out = engineer_features(frame())
    assert list(out.columns) == MODEL_INPUT_COLUMNS
    assert len(out) == 1


def test_input_frame_is_not_mutated():
    """engineer_features must be pure — the API reuses request frames."""
    df = frame()
    before = df.copy()
    engineer_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_zero_tenure_does_not_divide_by_zero():
    out = engineer_features(frame(tenure=0, total_charges=None, monthly_charges=45.3))
    assert out["avg_monthly_spend_ratio"].iloc[0] == pytest.approx(45.3)
    assert out["charge_trend_delta"].iloc[0] == pytest.approx(0.0)
    assert out.notna().all().all()


def test_blank_total_charges_becomes_zero_not_the_column_mean():
    """Imputing the mean would invent ~$2,280 of billing history for a
    brand-new customer and make them look loyal to the model."""
    out = engineer_features(frame(tenure=0, total_charges=None))
    assert out["total_charges"].iloc[0] == 0.0


def test_total_services_counts_only_active_addons():
    none_held = engineer_features(frame())
    assert none_held["total_services"].iloc[0] == 0

    all_held = engineer_features(frame(
        online_security="Yes", online_backup="Yes", device_protection="Yes",
        tech_support="Yes", streaming_tv="Yes", streaming_movies="Yes"))
    assert all_held["total_services"].iloc[0] == 6
    assert all_held["protection_services"].iloc[0] == 4


def test_no_internet_service_is_not_counted_as_a_subscription():
    out = engineer_features(frame(
        internet_service="No", online_security="No internet service",
        tech_support="No internet service"))
    assert out["total_services"].iloc[0] == 0
    assert out["no_protection_flag"].iloc[0] == 0   # cannot lack what is unsellable


def test_new_customer_risk_flag_needs_all_three_conditions():
    assert engineer_features(frame(tenure=3))["new_customer_risk_flag"].iloc[0] == 1
    assert engineer_features(frame(tenure=30))["new_customer_risk_flag"].iloc[0] == 0
    assert engineer_features(frame(tenure=3, contract="Two year"))[
        "new_customer_risk_flag"].iloc[0] == 0
    assert engineer_features(frame(tenure=3, tech_support="Yes"))[
        "new_customer_risk_flag"].iloc[0] == 0


def test_contract_ordinal_preserves_commitment_order():
    order = [engineer_features(frame(contract=c))["contract_ord"].iloc[0]
             for c in ("Month-to-month", "One year", "Two year")]
    assert order == [0, 1, 2]


def test_charge_trend_delta_detects_a_recent_price_rise():
    """Paid $50 for 20 months, now billed $90 — a $40 recent increase."""
    out = engineer_features(frame(tenure=20, total_charges=1000.0, monthly_charges=90.0))
    assert out["avg_monthly_spend_ratio"].iloc[0] == pytest.approx(50.0)
    assert out["charge_trend_delta"].iloc[0] == pytest.approx(40.0)


def test_no_feature_encodes_the_target():
    """Leakage guard: the churn label must not survive into the matrix."""
    out = engineer_features(frame())
    assert "churn" not in out.columns
    assert "churn_flag" not in out.columns


@pytest.mark.skipif(not DB_PATH.exists(), reason="run src/load_to_db.py first")
def test_runs_clean_over_the_whole_database():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM v_customer_360", conn)
    out = engineer_features(df)
    assert len(out) == len(df) == 7043
    assert out.isna().sum().sum() == 0
    assert out["total_services"].between(0, 6).all()

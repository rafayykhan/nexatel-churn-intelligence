"""
Phase 3 — feature engineering.

This module is deliberately the *only* place features are defined. The
training pipeline and the live API both import `engineer_features()`, so
a customer scored in the browser goes through byte-identical logic to the
rows the model was fitted on. Duplicating this logic in the backend is
the most common way a churn model quietly degrades in production.

Leakage statement
-----------------
Every feature below is computable from information available *before* a
customer cancels: demographics, contract terms, subscribed products, and
billing to date. Nothing reads `churn`, and nothing uses an aggregate
computed over the label (no target encoding, no churn-rate-by-segment
features). `total_charges` is historical billing, not a post-cancellation
settlement figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- raw column groups -------------------------------------------------
RAW_NUMERIC = ["tenure", "monthly_charges", "total_charges"]

RAW_BINARY = ["gender", "partner", "dependents", "phone_service",
              "paperless_billing"]

RAW_CATEGORICAL = ["multiple_lines", "internet_service", "online_security",
                   "online_backup", "device_protection", "tech_support",
                   "streaming_tv", "streaming_movies", "payment_method"]

ADDON_SERVICES = ["online_security", "online_backup", "device_protection",
                  "tech_support", "streaming_tv", "streaming_movies"]

PROTECTION_SERVICES = ["online_security", "online_backup",
                       "device_protection", "tech_support"]

MANUAL_PAYMENTS = {"Electronic check", "Mailed check"}

# Contract genuinely has an order (commitment length), so it is encoded
# ordinally rather than one-hot — it preserves "longer = stickier" for
# linear models instead of scattering it across three unordered dummies.
CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}

TENURE_BINS = [-0.1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]

# --- engineered columns, in the order the model expects ----------------
ENGINEERED_NUMERIC = [
    "tenure", "monthly_charges", "total_charges",
    "total_services", "protection_services",
    "avg_monthly_spend_ratio", "charge_trend_delta",
    "contract_ord", "tenure_group_ord",
    "senior_citizen", "new_customer_risk_flag",
    "manual_payment_flag", "no_protection_flag", "is_fiber",
]

ENGINEERED_CATEGORICAL = RAW_BINARY + RAW_CATEGORICAL + ["contract", "tenure_group"]

MODEL_INPUT_COLUMNS = ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL


def _yes(series: pd.Series) -> pd.Series:
    """1 where the service is actively subscribed.

    'No internet service' is *not* the same as 'No' — the first means the
    add-on was never purchasable, the second means it was declined. Both
    map to 0 here for counting purposes, but the distinction survives in
    the one-hot encoding of the raw column.
    """
    return (series.astype(str).str.strip() == "Yes").astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer rows -> model-ready feature frame.

    Accepts the snake_case schema returned by `v_customer_360` (or a
    single-row frame built from an API request). Pure and side-effect
    free: the input frame is never mutated.
    """
    out = df.copy()

    # ---- 1. data quality fix from EDA --------------------------------
    # total_charges is blank for the 11 customers with tenure = 0: they
    # have been signed up but never billed. Imputing the column mean
    # would invent ~$2,280 of history for a brand-new customer and make
    # them look loyal. 0 is the factually correct value.
    out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce")
    out["total_charges"] = out["total_charges"].fillna(0.0)
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce").fillna(0).astype(int)
    out["monthly_charges"] = pd.to_numeric(out["monthly_charges"], errors="coerce").fillna(0.0)
    out["senior_citizen"] = pd.to_numeric(
        out.get("senior_citizen", 0), errors="coerce").fillna(0).astype(int)

    # ---- 2. product depth -------------------------------------------
    # Hypothesis from EDA Q10: churn falls monotonically from 45.8% at one
    # add-on to 5.3% at six. Each extra product is another switching cost.
    out["total_services"] = sum(_yes(out[c]) for c in ADDON_SERVICES)

    # Protection add-ons only (security/backup/device/support), excluding
    # streaming. Streaming is entertainment; protection is the sticky,
    # support-shaped stuff — Q08 shows 41.6% vs 15.2% churn on tech
    # support alone, so the two groups deserve separate signals.
    out["protection_services"] = sum(_yes(out[c]) for c in PROTECTION_SERVICES)

    # ---- 3. spend behaviour -----------------------------------------
    # Historical average monthly spend. Guarded against tenure = 0.
    safe_tenure = out["tenure"].clip(lower=1)
    out["avg_monthly_spend_ratio"] = (out["total_charges"] / safe_tenure).round(4)
    # For never-billed customers the historical average is undefined; the
    # best available estimate of their spend is their current plan price.
    out.loc[out["tenure"] == 0, "avg_monthly_spend_ratio"] = out.loc[
        out["tenure"] == 0, "monthly_charges"]

    # Current price minus historical average. Positive = this customer is
    # paying more now than they have on average — a recent upsell or price
    # rise, both of which precede cancellation calls.
    out["charge_trend_delta"] = (
        out["monthly_charges"] - out["avg_monthly_spend_ratio"]).round(4)

    # ---- 4. ordinal encodings ---------------------------------------
    out["contract_ord"] = out["contract"].map(CONTRACT_ORDER).fillna(0).astype(int)
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_BINS,
                                 labels=TENURE_LABELS).astype(str)
    out["tenure_group_ord"] = out["tenure_group"].map(
        {lab: i for i, lab in enumerate(TENURE_LABELS)}).fillna(0).astype(int)

    # ---- 5. business risk flags -------------------------------------
    # The Q09 segment, encoded directly: 66.7% churn, 904 customers.
    # Tree models can in principle discover this interaction themselves;
    # handing it over explicitly helps the linear baseline and makes the
    # SHAP output legible to a retention agent.
    out["new_customer_risk_flag"] = (
        (out["tenure"] < 6)
        & (out["contract"] == "Month-to-month")
        & (out["tech_support"].astype(str) == "No")
    ).astype(int)

    # Manual payment = a monthly re-decision to keep paying. Q06: 45.3%
    # churn on electronic check vs 15.2% on automatic credit card.
    out["manual_payment_flag"] = out["payment_method"].isin(MANUAL_PAYMENTS).astype(int)

    # Internet customer holding zero protection add-ons.
    out["no_protection_flag"] = (
        (out["internet_service"].astype(str) != "No") & (out["protection_services"] == 0)
    ).astype(int)

    # Fiber carries a 41.9% churn rate at a $91.50 average price (Q04) —
    # the premium-price/expectation-gap segment.
    out["is_fiber"] = (out["internet_service"].astype(str) == "Fiber optic").astype(int)

    return out[MODEL_INPUT_COLUMNS]


FEATURE_JUSTIFICATIONS = [
    ("total_services",
     "Count of the six optional add-ons. Each additional product is a switching "
     "cost. EDA Q10 shows churn falling from 45.8% (one add-on) to 5.3% (six)."),
    ("protection_services",
     "Security/backup/device-protection/tech-support only, excluding streaming. "
     "Support-shaped products correlate with retention far more strongly than "
     "entertainment add-ons (41.6% vs 15.2% churn on tech support alone)."),
    ("avg_monthly_spend_ratio",
     "TotalCharges / tenure — historical average spend. Separates a customer who "
     "has always paid $90 from one recently upgraded to $90. Guarded for tenure=0, "
     "where the current plan price is substituted."),
    ("charge_trend_delta",
     "monthly_charges minus the historical average. Positive values flag a recent "
     "upsell or price rise, which is what triggers a cancellation call."),
    ("contract_ord",
     "Ordinal 0/1/2 for month-to-month/one-year/two-year. Contract length is "
     "genuinely ordered, so one-hot encoding would discard the ordering that "
     "drives the strongest single effect in the data (42.7% vs 2.8% churn)."),
    ("tenure_group / tenure_group_ord",
     "Lifecycle buckets 0-12, 13-24, 25-48, 49+. Churn is 47.4% in year one and "
     "9.5% after four years — the relationship is a steep decay, not a straight "
     "line, and bucketing lets linear models capture that."),
    ("new_customer_risk_flag",
     "tenure < 6 AND month-to-month AND no tech support. This exact intersection "
     "churns at 66.7% across 904 customers. Encoding it explicitly gives the "
     "linear baseline the interaction and makes SHAP output readable to an agent."),
    ("manual_payment_flag",
     "Electronic or mailed check. A manual payment is a monthly re-decision to "
     "stay: 45.3% churn vs 15.2% on automatic credit card."),
    ("no_protection_flag",
     "Internet subscriber holding zero protection add-ons — an unprotected, "
     "unsupported customer whose first bad experience has no safety net."),
    ("is_fiber",
     "Fiber churns at 41.9% on a $91.50 average bill versus 19.0% for DSL: a "
     "premium-price expectation gap, not a general internet effect."),
]

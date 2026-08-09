"""
Phase 6 — model explainability.

Two jobs:
  1. Global: which features drive churn across the whole book (SHAP
     summary + bar plots, written to reports/figures/).
  2. Local: for one customer, the top factors pushing them toward or
     away from cancelling — translated into sentences a retention agent
     can actually say on a call, plus a suggested action.

`explain_customer()` is imported directly by the FastAPI backend, so the
explanation shown in the browser is the same computation audited here.

Run:  python src/explain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from config import DB_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from features import ENGINEERED_NUMERIC, engineer_features  # noqa: E402

# Human-readable names for the encoded model-matrix columns. Anything not
# listed falls back to a de-underscored version of the raw column name.
PRETTY = {
    "tenure": "months as a customer",
    "monthly_charges": "current monthly bill",
    "total_charges": "total billed to date",
    "total_services": "number of add-on services",
    "protection_services": "protection add-ons held",
    "avg_monthly_spend_ratio": "historical average monthly spend",
    "charge_trend_delta": "recent bill increase vs. their own average",
    "contract_ord": "contract length",
    "tenure_group_ord": "lifecycle stage",
    "senior_citizen": "senior citizen",
    "new_customer_risk_flag": "new + month-to-month + no tech support",
    "manual_payment_flag": "pays manually (check)",
    "no_protection_flag": "no protection add-ons",
    "is_fiber": "fiber optic internet",
    "contract_Month-to-month": "month-to-month contract",
    "contract_One year": "one-year contract",
    "contract_Two year": "two-year contract",
    "internet_service_Fiber optic": "fiber optic internet",
    "internet_service_DSL": "DSL internet",
    "internet_service_No": "no internet service",
    "payment_method_Electronic check": "pays by electronic check",
    "payment_method_Mailed check": "pays by mailed check",
    "payment_method_Bank transfer (automatic)": "automatic bank transfer",
    "payment_method_Credit card (automatic)": "automatic credit card",
    "tech_support_No": "no tech support add-on",
    "tech_support_Yes": "has tech support",
    "online_security_No": "no online security",
    "online_security_Yes": "has online security",
    "paperless_billing_Yes": "paperless billing",
    "dependents_Yes": "has dependents on account",
    "partner_Yes": "has a partner on account",
    "tenure_group_0-12": "first year of service",
    "tenure_group_49+": "long-tenured customer",
}

# Numeric columns that are really booleans. A flag sitting at 0 carries a
# real SHAP value, but "no protection add-ons: 0" is not a sentence an
# agent can use, so they are filtered out the same way inactive one-hot
# dummies are.
FLAG_FEATURES = {"senior_citizen", "new_customer_risk_flag", "manual_payment_flag",
                 "no_protection_flag", "is_fiber"}

# Several columns encode the same underlying fact — contract_ord and
# contract_Month-to-month are one concept split across an ordinal and a
# dummy. Reasons are deduplicated by concept, not by column name, so the
# agent sees "month-to-month contract" once instead of three near-copies.
CONCEPT_PREFIXES = {
    "contract": "contract", "contract_ord": "contract",
    "internet_service": "internet_service", "is_fiber": "internet_service",
    "tenure_group": "tenure", "tenure_group_ord": "tenure", "tenure": "tenure",
    "payment_method": "payment_method", "manual_payment_flag": "payment_method",
    "tech_support": "tech_support",
    "total_services": "product_depth", "protection_services": "product_depth",
    "no_protection_flag": "product_depth",
    "monthly_charges": "spend", "avg_monthly_spend_ratio": "spend",
    "total_charges": "spend", "charge_trend_delta": "spend",
}


def _concept(name: str) -> str:
    """Map a model-matrix column to the business fact it represents."""
    if name in CONCEPT_PREFIXES:
        return CONCEPT_PREFIXES[name]
    base = name.split("_")[0] if "_" not in name else name.rsplit("_", 1)[0]
    return CONCEPT_PREFIXES.get(base, name)


# Retention playbook — first matching rule wins. Keyed on the drivers the
# model surfaces most often, so the action lines up with the reason.
PLAYBOOK = [
    (lambda c: c["contract"] == "Month-to-month" and c["tenure"] < 12,
     "Offer a 12-month contract with the first two months discounted — "
     "contract length is the single largest retention lever in the data "
     "(42.7% churn month-to-month vs 2.8% on two-year)."),
    (lambda c: c["tech_support"] == "No" and c["internet_service"] != "No",
     "Bundle tech support free for six months. Internet customers without "
     "it churn at 41.6% versus 15.2% with it."),
    (lambda c: c["payment_method"] in ("Electronic check", "Mailed check"),
     "Move them to automatic card or bank payment with a small credit — "
     "manual payers churn at 45.3% vs 15.2% on autopay."),
    (lambda c: c["internet_service"] == "Fiber optic" and c["monthly_charges"] > 80,
     "Fiber customers above $80/month churn at 41.9%. Run a line-quality "
     "check and offer a plan review before they price-shop."),
    (lambda c: c["total_services"] <= 1,
     "Low product depth. Add one protection add-on at no cost — churn "
     "falls from 45.8% at one add-on to 12.4% at five."),
    (lambda c: True,
     "No single dominant driver. Log a courtesy check-in call and confirm "
     "the account details are current."),
]


def load_pipeline():
    return joblib.load(MODELS_DIR / "churn_pipeline.pkl")


def load_metadata() -> dict:
    return json.loads((MODELS_DIR / "model_metadata.json").read_text())


def _prettify(name: str) -> str:
    """Model-matrix column -> a phrase an agent can read aloud.

    The explicit PRETTY map covers the columns that actually surface as
    top drivers; the fallback handles the long tail of one-hot dummies so
    a rare driver still renders as "no streaming tv" rather than
    "streaming_tv_No".
    """
    if name in PRETTY:
        return PRETTY[name]
    for suffix, template in (("_No internet service", "{} (not available)"),
                             ("_No phone service", "{} (not available)"),
                             ("_Yes", "has {}"),
                             ("_No", "no {}")):
        if name.endswith(suffix):
            return template.format(name[: -len(suffix)].replace("_", " "))
    return name.replace("_", " ")


def get_explainer(pipeline):
    """SHAP explainer matched to the final estimator type.

    Tree models get TreeExplainer (exact, fast, no background sample).
    Linear models get LinearExplainer. Anything else falls back to
    KernelExplainer on a small background set.
    """
    import shap

    model = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]
    kind = type(model).__name__

    if kind in {"RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier",
                "ExtraTreesClassifier", "LGBMClassifier"}:
        return shap.TreeExplainer(model), pre, "tree"
    if kind == "LogisticRegression":
        bg = _background_matrix(pre)
        return shap.LinearExplainer(model, bg), pre, "linear"
    bg = shap.kmeans(_background_matrix(pre), 25)
    return shap.KernelExplainer(model.predict_proba, bg), pre, "kernel"


def _background_matrix(pre) -> np.ndarray:
    """A small transformed sample of real customers, for explainers that
    need a reference distribution."""
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM v_customer_360", conn)
    sample = df.sample(min(300, len(df)), random_state=RANDOM_STATE)
    return pre.transform(engineer_features(sample))


def shap_values_for(explainer, matrix: np.ndarray, kind: str) -> np.ndarray:
    """Return a 1-D array of SHAP values for the positive class."""
    vals = explainer.shap_values(matrix)
    if isinstance(vals, list):          # older API: one array per class
        vals = vals[1] if len(vals) > 1 else vals[0]
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # (rows, features, classes)
        vals = vals[:, :, -1]
    return vals


def explain_customer(customer: dict, pipeline, explainer, pre, kind: str,
                     top_n: int = 3) -> dict:
    """Score one customer and return probability + ranked human reasons."""
    row = pd.DataFrame([customer])
    feats = engineer_features(row)
    proba = float(pipeline.predict_proba(feats)[:, 1][0])

    matrix = pre.transform(feats)
    names = list(pre.get_feature_names_out())
    values = shap_values_for(explainer, matrix, kind)[0]
    encoded = np.asarray(matrix)[0]

    order = np.argsort(np.abs(values))[::-1]
    risk_factors, protective_factors = [], []
    seen: set[str] = set()

    for idx in order:
        name = names[idx]
        label = _prettify(name)

        # A one-hot column sitting at 0 means the customer does NOT have
        # that attribute. Its SHAP value is real, but showing an agent
        # "fiber optic internet" for a DSL customer is worse than showing
        # nothing, so inactive dummies are dropped from the narrative.
        # Activity has to be judged on the engineered value, not on the
        # transformed matrix: StandardScaler maps a flag of 0 to a
        # non-zero z-score, so testing the scaled column would never
        # filter anything out. One-hot columns are not scaled, so those
        # are read straight off the matrix.
        if name in ENGINEERED_NUMERIC:
            if name in FLAG_FEATURES and float(feats.iloc[0][name]) == 0:
                continue
        elif encoded[idx] == 0:
            continue

        # Keep the strongest column per business concept. We walk in
        # descending |SHAP|, so the first one seen is the strongest;
        # everything else describing the same fact is dropped rather than
        # repeated back to the agent in three different wordings.
        concept = _concept(name)
        if concept in seen:
            continue
        seen.add(concept)

        entry = {
            "feature": name,
            "label": label,
            "impact": round(float(values[idx]), 4),
            "value": _display_value(name, feats.iloc[0], float(encoded[idx])),
        }
        if values[idx] > 0 and len(risk_factors) < top_n:
            risk_factors.append(entry)
        elif values[idx] < 0 and len(protective_factors) < top_n:
            protective_factors.append(entry)
        if len(risk_factors) >= top_n and len(protective_factors) >= top_n:
            break

    enriched = dict(customer)
    enriched["total_services"] = int(feats.iloc[0]["total_services"])
    action = next(msg for cond, msg in PLAYBOOK if cond(enriched))

    return {
        "churn_probability": round(proba, 4),
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "recommended_action": action,
    }


def _display_value(encoded_name: str, feature_row: pd.Series, encoded_value: float):
    """Readable value for a model-matrix column.

    Numeric features report the engineered value the customer actually has
    (pre-scaling, so '3 months' not '-1.28 standard deviations'). One-hot
    columns only reach here when they fired, so they report 'yes'.
    """
    if encoded_name in feature_row.index:
        v = feature_row[encoded_name]
        return round(float(v), 2) if isinstance(v, (int, float, np.number)) else str(v)
    return "yes"


# ---------------------------------------------------------------------
def main() -> None:
    import shap

    pipeline = load_pipeline()
    meta = load_metadata()
    explainer, pre, kind = get_explainer(pipeline)
    print(f"final model : {meta['final_model']}  (explainer: {kind})")

    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM v_customer_360", conn)

    sample = df.sample(1200, random_state=RANDOM_STATE)
    feats = engineer_features(sample)
    matrix = pre.transform(feats)
    names = list(pre.get_feature_names_out())
    values = shap_values_for(explainer, matrix, kind)
    print(f"computed SHAP values for {values.shape[0]} customers x {values.shape[1]} features")

    pretty_names = [_prettify(n) for n in names]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, matrix, feature_names=pretty_names, show=False,
                      max_display=15, plot_size=None)
    plt.title("SHAP — how each factor moves churn risk", fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  figure -> 10_shap_summary.png")

    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, matrix, feature_names=pretty_names, plot_type="bar",
                      show=False, max_display=15, plot_size=None)
    plt.title("Global feature importance (mean |SHAP|)", fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_shap_importance.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  figure -> 11_shap_importance.png")

    importance = (pd.DataFrame({"feature": pretty_names,
                                "mean_abs_shap": np.abs(values).mean(axis=0)})
                  .groupby("feature", as_index=False)["mean_abs_shap"].max()
                  .sort_values("mean_abs_shap", ascending=False)
                  .head(15))
    importance["mean_abs_shap"] = importance["mean_abs_shap"].round(4)
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    print("\nTop global drivers")
    print(importance.to_string(index=False))

    # worked example: a high-risk customer from the real data
    demo = df[(df.contract == "Month-to-month") & (df.tenure < 6)
              & (df.tech_support == "No") & (df.churn == "Yes")].iloc[0].to_dict()
    result = explain_customer(demo, pipeline, explainer, pre, kind)
    print(f"\nworked example — customer {demo['customer_id']} "
          f"(actual outcome: churn={demo['churn']})")
    print(f"  predicted risk: {result['churn_probability']:.1%}")
    for r in result["risk_factors"]:
        print(f"    + {r['label']:<45} impact {r['impact']:+.3f}")
    for p in result["protective_factors"]:
        print(f"    - {p['label']:<45} impact {p['impact']:+.3f}")
    print(f"  action: {result['recommended_action'][:80]}...")

    (REPORTS_DIR / "explanation_example.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

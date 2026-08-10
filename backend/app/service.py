"""
Model service layer.

Artifacts are loaded exactly once at import and reused for every request
— joblib.load on each call would add ~200ms and defeat the point of a
warm process. Feature engineering and explanation are imported from
`src/`, the same modules the training run used.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

# Repo root -> import the shared feature/explain modules
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import explain as explain_mod  # noqa: E402
from config import EXPECTED_LIFETIME_MONTHS, MODELS_DIR, REPORTS_DIR  # noqa: E402

# The medium band floor is derived from the operating threshold rather than
# hardcoded. Tuning moved the threshold to 0.31, below the old fixed 0.35
# floor, which made "Medium" unreachable — every customer came back High or
# Low. Anchoring it at 60% of the threshold keeps the three bands ordered
# whatever the threshold retunes to on the next training run.
MEDIUM_BAND_RATIO = 0.6


class ChurnService:
    def __init__(self) -> None:
        self.pipeline = explain_mod.load_pipeline()
        self.metadata = explain_mod.load_metadata()
        self.threshold: float = float(self.metadata["decision_threshold"])
        self.model_name: str = self.metadata["final_model"]
        try:
            self.explainer, self.pre, self.kind = explain_mod.get_explainer(self.pipeline)
            self.explanations_enabled = True
        except Exception as exc:                      # pragma: no cover
            # A missing SHAP wheel must degrade the response, not break it.
            print(f"[warn] SHAP explainer unavailable, falling back to rules: {exc}")
            self.explainer = self.pre = self.kind = None
            self.explanations_enabled = False

    # -- risk banding ---------------------------------------------------
    @property
    def medium_floor(self) -> float:
        return round(self.threshold * MEDIUM_BAND_RATIO, 3)

    def band(self, proba: float) -> str:
        """High = above the outreach threshold, so the agent must act.

        Medium is a genuine watch-list, not a rounding of Low: these are
        customers the model puts materially above baseline but below the
        point where an offer pays for itself.
        """
        if proba >= self.threshold:
            return "High"
        if proba >= self.medium_floor:
            return "Medium"
        return "Low"

    # -- scoring --------------------------------------------------------
    def predict(self, customer: dict) -> dict:
        payload = dict(customer)
        # total_charges is optional on the wire: if the agent doesn't have
        # it, estimate from the plan price and tenure rather than reject.
        if payload.get("total_charges") is None:
            payload["total_charges"] = round(
                float(payload["monthly_charges"]) * int(payload["tenure"]), 2)

        if self.explanations_enabled:
            result = explain_mod.explain_customer(
                payload, self.pipeline, self.explainer, self.pre, self.kind)
        else:
            import pandas as pd
            from features import engineer_features
            feats = engineer_features(pd.DataFrame([payload]))
            proba = float(self.pipeline.predict_proba(feats)[:, 1][0])
            enriched = dict(payload)
            enriched["total_services"] = int(feats.iloc[0]["total_services"])
            action = next(msg for cond, msg in explain_mod.PLAYBOOK if cond(enriched))
            result = {"churn_probability": round(proba, 4), "risk_factors": [],
                      "protective_factors": [], "recommended_action": action}

        proba = result["churn_probability"]
        return {
            **result,
            "risk_level": self.band(proba),
            "risk_score": int(round(proba * 100)),
            "decision_threshold": self.threshold,
            "flagged_for_outreach": proba >= self.threshold,
            "revenue_at_risk_annual": round(
                float(payload["monthly_charges"]) * EXPECTED_LIFETIME_MONTHS, 2),
            "model_name": self.model_name,
        }

    def predict_batch(self, customers: list[dict]) -> dict:
        results = [self.predict(c) for c in customers]
        flagged = [r for r in results if r["flagged_for_outreach"]]
        return {
            "results": results,
            "summary": {
                "scored": len(results),
                "flagged_for_outreach": len(flagged),
                "high_risk": sum(r["risk_level"] == "High" for r in results),
                "medium_risk": sum(r["risk_level"] == "Medium" for r in results),
                "low_risk": sum(r["risk_level"] == "Low" for r in results),
                "annual_revenue_at_risk": round(
                    sum(r["revenue_at_risk_annual"] for r in flagged), 2),
            },
        }

    # -- supporting reads ----------------------------------------------
    def model_info(self) -> dict:
        biz = self.metadata["metrics_at_business_threshold"]
        return {
            "model": self.model_name,
            "decision_threshold": self.threshold,
            "explanations_enabled": self.explanations_enabled,
            "trained_rows": self.metadata["trained_rows"],
            "test_rows": self.metadata["test_rows"],
            "metrics": {k: biz[k] for k in
                        ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
            "confusion_matrix": biz["confusion_matrix"],
            "metrics_at_default_threshold": self.metadata["metrics_at_default_threshold"],
            "economics": self.metadata["economics"],
            "best_params": self.metadata["best_params"],
        }


@lru_cache(maxsize=1)
def get_service() -> ChurnService:
    return ChurnService()


@lru_cache(maxsize=1)
def get_eda_stats() -> dict:
    path = REPORTS_DIR / "eda_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache(maxsize=1)
def get_shap_importance() -> list[dict]:
    import csv
    path = REPORTS_DIR / "shap_importance.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return [{"feature": r["feature"], "importance": float(r["mean_abs_shap"])}
                for r in csv.DictReader(fh)]


@lru_cache(maxsize=1)
def get_model_comparison() -> list[dict]:
    import csv
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))

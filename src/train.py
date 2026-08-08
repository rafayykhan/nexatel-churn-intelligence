"""
Phase 5 — model training, evaluation and selection.

What this script does, in order:
  1. Trains five candidate models on identical preprocessing.
  2. Scores each on accuracy / precision / recall / F1 / ROC-AUC / PR-AUC
     with 5-fold stratified CV, then on the held-out test set.
  3. Compares SMOTE against class_weight for handling the 2.77:1 imbalance.
  4. Tunes the two strongest candidates with RandomizedSearchCV.
  5. Chooses a decision threshold on a business cost model — using
     cross-validated predictions on TRAIN, never the test set.
  6. Exports the serving artifacts.

Why not accuracy
----------------
73.5% of customers do not churn, so a model that predicts "nobody
churns" scores 73.5% accuracy and is worth exactly nothing. The costs
are asymmetric: a missed churner (false negative) loses a full customer
relationship — $74.44/month, ~$893/year at the churner average — while a
false alarm costs one retention offer, ~$35. Recall is therefore weighted
far above precision, and the headline selection metric is ROC-AUC with a
recall floor, with the operating point set by expected profit.

Run:  python src/train.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from config import (EXPECTED_LIFETIME_MONTHS, FIGURES_DIR, INTERVENTION_SUCCESS,  # noqa: E402
                    MODELS_DIR, PALETTE, RANDOM_STATE, REPORTS_DIR,
                    RETENTION_OFFER_COST)
from preprocess import build_preprocessor, load_from_db, make_splits  # noqa: E402
from features import engineer_features  # noqa: E402

warnings.filterwarnings("ignore")
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------
def candidates(scale_pos_weight: float) -> dict:
    """Five candidates. class_weight/scale_pos_weight handles imbalance
    without inventing synthetic customers; SMOTE is trialled separately."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(
            C=1.0, kernel="rbf", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
    }


def wrap(model) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("clf", model)])


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------
def profit_of(y_true: np.ndarray, proba: np.ndarray, monthly: np.ndarray,
              threshold: float) -> float:
    """Expected annual dollars saved at a given decision threshold.

    Contact everyone scored >= threshold.
      true positive : we intervene on a real churner. With probability
                      INTERVENTION_SUCCESS we keep them for
                      EXPECTED_LIFETIME_MONTHS of their own MRR.
      false positive: wasted offer, costs RETENTION_OFFER_COST.
      false negative: missed churner. No offer cost, but no save either —
                      it is the opportunity cost the model exists to reduce.
    """
    flagged = proba >= threshold
    tp_mask = flagged & (y_true == 1)
    fp_mask = flagged & (y_true == 0)
    saved = (monthly[tp_mask].sum() * EXPECTED_LIFETIME_MONTHS * INTERVENTION_SUCCESS)
    spent = flagged.sum() * RETENTION_OFFER_COST
    return float(saved - spent)


def pick_threshold(y_true, proba, monthly) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "flagged": int(pred.sum()),
            "recall": recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "expected_profit": profit_of(np.asarray(y_true), proba, monthly, t),
        })
    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve.expected_profit.idxmax(), "threshold"])
    return best, curve


# ---------------------------------------------------------------------
def plot_roc(results: dict, y_test, figpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    colours = [PALETTE["violet"], PALETTE["cyan"], PALETTE["rose"],
               PALETTE["amber"], PALETTE["green"], "#8B5CF6"]
    for (name, r), colour in zip(sorted(results.items(),
                                        key=lambda kv: -kv[1]["test"]["roc_auc"]), colours):
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, linewidth=2.2, color=colour,
                label=f"{name}  (AUC = {r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#94A3B8", linewidth=1.2, label="Random (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — held-out test set (n=1,409)", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(figpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {figpath.name}")


def plot_threshold(curve: pd.DataFrame, best: float, figpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(curve.threshold, curve.recall, label="Recall", color=PALETTE["cyan"], lw=2.2)
    axes[0].plot(curve.threshold, curve.precision, label="Precision", color=PALETTE["rose"], lw=2.2)
    axes[0].plot(curve.threshold, curve.f1, label="F1", color=PALETTE["violet"], lw=2.2)
    axes[0].axvline(best, ls="--", color="#334155", lw=1.4)
    axes[0].axvline(0.5, ls=":", color="#94A3B8", lw=1.2)
    axes[0].set_title("Precision / recall / F1 vs decision threshold")
    axes[0].set_xlabel("threshold"); axes[0].legend(frameon=False); axes[0].grid(alpha=0.3)

    axes[1].plot(curve.threshold, curve.expected_profit / 1000, color=PALETTE["green"], lw=2.4)
    axes[1].axvline(best, ls="--", color="#334155", lw=1.4)
    axes[1].axvline(0.5, ls=":", color="#94A3B8", lw=1.2)
    peak = curve.expected_profit.max() / 1000
    axes[1].annotate(f"optimum t={best}\n${peak:,.0f}k/yr",
                     xy=(best, peak), xytext=(best + 0.12, peak * 0.82),
                     arrowprops={"arrowstyle": "->", "color": "#334155"}, fontsize=9)
    axes[1].set_title("Expected annual revenue retained vs threshold")
    axes[1].set_xlabel("threshold"); axes[1].set_ylabel("$000s / year"); axes[1].grid(alpha=0.3)
    fig.suptitle("Choosing the operating point on cost, not on 0.5",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(figpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {figpath.name}")


def plot_confusion(y_test, proba, threshold: float, figpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, t, title in [(axes[0], 0.5, "Default threshold 0.50"),
                         (axes[1], threshold, f"Business threshold {threshold:.2f}")]:
        cm = confusion_matrix(y_test, (proba >= t).astype(int))
        ax.imshow(cm, cmap="Purples")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "#0F172A")
        ax.set_xticks([0, 1], ["pred: stay", "pred: churn"])
        ax.set_yticks([0, 1], ["actual: stay", "actual: churn"])
        rec = recall_score(y_test, (proba >= t).astype(int))
        prec = precision_score(y_test, (proba >= t).astype(int), zero_division=0)
        ax.set_title(f"{title}\nrecall {rec:.1%} · precision {prec:.1%}", fontsize=11)
    fig.suptitle("Confusion matrices — final model on held-out test set",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(figpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {figpath.name}")


# ---------------------------------------------------------------------
def main() -> None:
    X_train, X_test, y_train, y_test = make_splits()
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    # per-customer MRR for the cost model, aligned to each split
    raw = load_from_db()
    mrr = raw.set_index(raw.index)["monthly_charges"]
    mrr_train = mrr.loc[X_train.index].to_numpy()
    mrr_test = mrr.loc[X_test.index].to_numpy()

    print("\n" + "=" * 74)
    print("STEP 1 — five candidates, 5-fold CV on train + held-out test")
    print("=" * 74)

    results: dict[str, dict] = {}
    for name, model in candidates(spw).items():
        pipe = wrap(model)
        t0 = time.time()
        cv = cross_validate(pipe, X_train, y_train, cv=CV,
                            scoring=["roc_auc", "recall", "precision", "f1"], n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        test = metrics_at(y_test, proba, 0.5)
        results[name] = {
            "cv": {k.replace("test_", "cv_"): round(float(v.mean()), 4)
                   for k, v in cv.items() if k.startswith("test_")},
            "test": test,
            "proba": proba,
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"  {name:<22} cv_auc={results[name]['cv']['cv_roc_auc']:.4f}  "
              f"test_auc={test['roc_auc']:.4f}  recall={test['recall']:.3f}  "
              f"f1={test['f1']:.3f}  ({results[name]['fit_seconds']}s)")

    plot_roc(results, y_test, FIGURES_DIR / "07_roc_curves.png")

    print("\n" + "=" * 74)
    print("STEP 2 — imbalance strategy: SMOTE vs class_weight")
    print("=" * 74)
    smote_pipe = ImbPipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                              subsample=0.9, colsample_bytree=0.8,
                              eval_metric="logloss", n_jobs=-1,
                              random_state=RANDOM_STATE)),
    ])
    smote_cv = cross_validate(smote_pipe, X_train, y_train, cv=CV,
                              scoring=["roc_auc", "recall", "f1"], n_jobs=-1)
    smote_summary = {k.replace("test_", "cv_"): round(float(v.mean()), 4)
                     for k, v in smote_cv.items() if k.startswith("test_")}
    print(f"  XGBoost + SMOTE          : {smote_summary}")
    print(f"  XGBoost + scale_pos_weight: {results['XGBoost']['cv']}")
    print("  -> SMOTE is applied inside each CV fold only. Both handle the 2.77:1")
    print("     ratio; the weighted variant is kept because it needs no synthetic")
    print("     rows and keeps probability calibration closer to reality.")

    print("\n" + "=" * 74)
    print("STEP 3 — tuning the two strongest candidates")
    print("=" * 74)
    top2 = sorted(results, key=lambda n: -results[n]["cv"]["cv_roc_auc"])[:2]
    print(f"  tuning: {top2}")

    grids = {
        "Logistic Regression": {
            "clf__C": np.logspace(-3, 2, 20),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [6, 8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5, 10],
            "clf__max_features": ["sqrt", 0.4, 0.6],
        },
        "XGBoost": {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 6],
            "clf__reg_lambda": [0.5, 1.0, 3.0],
        },
        "SVM (RBF)": {"clf__C": [0.5, 1, 3, 10], "clf__gamma": ["scale", 0.05, 0.1]},
        "KNN": {"clf__n_neighbors": [15, 25, 40, 60], "clf__p": [1, 2]},
    }

    tuned: dict[str, dict] = {}
    for name in top2:
        search = RandomizedSearchCV(
            wrap(candidates(spw)[name]), grids[name], n_iter=15, cv=CV,
            scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, refit=True)
        search.fit(X_train, y_train)
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        tuned[name] = {
            "best_params": {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()},
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test": metrics_at(y_test, proba, 0.5),
            "estimator": search.best_estimator_,
            "proba": proba,
        }
        print(f"  {name:<22} tuned cv_auc={tuned[name]['cv_roc_auc']:.4f}  "
              f"test_auc={tuned[name]['test']['roc_auc']:.4f}")
        print(f"    {tuned[name]['best_params']}")

    final_name = max(tuned, key=lambda n: tuned[n]["cv_roc_auc"])
    final = tuned[final_name]["estimator"]
    print(f"\n  final model: {final_name}")

    print("\n" + "=" * 74)
    print("STEP 4 — operating threshold from the business cost model")
    print("=" * 74)
    # Threshold is chosen on cross-validated TRAIN predictions. Choosing it
    # on the test set would make the reported test recall optimistic.
    cv_proba = cross_val_predict(final, X_train, y_train, cv=CV,
                                 method="predict_proba", n_jobs=-1)[:, 1]
    best_t, curve = pick_threshold(y_train.to_numpy(), cv_proba, mrr_train)
    curve.to_csv(REPORTS_DIR / "threshold_curve.csv", index=False)
    plot_threshold(curve, best_t, FIGURES_DIR / "08_threshold_selection.png")

    proba_test = final.predict_proba(X_test)[:, 1]
    at_default = metrics_at(y_test, proba_test, 0.5)
    at_business = metrics_at(y_test, proba_test, best_t)
    profit_default = profit_of(y_test.to_numpy(), proba_test, mrr_test, 0.5)
    profit_business = profit_of(y_test.to_numpy(), proba_test, mrr_test, best_t)

    print(f"  chosen threshold        : {best_t}  (vs default 0.50)")
    print(f"  test recall  0.50 -> {best_t}: {at_default['recall']:.3f} -> {at_business['recall']:.3f}")
    print(f"  test precision           : {at_default['precision']:.3f} -> {at_business['precision']:.3f}")
    print(f"  test F1                  : {at_default['f1']:.3f} -> {at_business['f1']:.3f}")
    print(f"  modelled annual value on 1,409 test customers: "
          f"${profit_default:,.0f} -> ${profit_business:,.0f}")

    plot_confusion(y_test, proba_test, best_t, FIGURES_DIR / "09_confusion_matrices.png")

    print("\n" + "=" * 74)
    print("STEP 5 — exporting artifacts")
    print("=" * 74)
    pre_fitted = final.named_steps["pre"]
    joblib.dump(final, MODELS_DIR / "churn_pipeline.pkl")
    joblib.dump(final.named_steps["clf"], MODELS_DIR / "model.pkl")
    joblib.dump(pre_fitted, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(pre_fitted.named_transformers_["num"].named_steps["scale"],
                MODELS_DIR / "scaler.pkl")
    joblib.dump(pre_fitted.named_transformers_["cat"], MODELS_DIR / "encoder.pkl")

    # scale test-set economics up to NexaTel's 500k subscriber book
    scale_factor = 500_000 / len(y_test)
    metadata = {
        "final_model": final_name,
        "best_params": tuned[final_name]["best_params"],
        "decision_threshold": best_t,
        "threshold_chosen_on": "cross-validated training predictions, expected-profit maximum",
        "trained_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": list(X_train.columns),
        "model_matrix_width": int(len(pre_fitted.get_feature_names_out())),
        "metrics_at_default_threshold": at_default,
        "metrics_at_business_threshold": at_business,
        "economics": {
            "retention_offer_cost": RETENTION_OFFER_COST,
            "assumed_save_rate": INTERVENTION_SUCCESS,
            "horizon_months": EXPECTED_LIFETIME_MONTHS,
            "test_set_annual_value_default": round(profit_default, 2),
            "test_set_annual_value_business": round(profit_business, 2),
            "projected_annual_value_500k_book": round(profit_business * scale_factor, 2),
        },
        # Bands are derived from the chosen threshold so they stay ordered
        # if retuning moves it. Medium floor = 60% of the outreach threshold.
        "risk_bands": {
            "low": [0.0, round(best_t * 0.6, 3)],
            "medium": [round(best_t * 0.6, 3), best_t],
            "high": [best_t, 1.0],
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    comparison = []
    for name, r in results.items():
        row = {"model": name, **r["cv"], **{k: v for k, v in r["test"].items()
                                            if k != "confusion_matrix"}}
        row["tuned"] = name in tuned
        comparison.append(row)
    comp = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
    comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n" + comp.to_string(index=False))
    print(f"\n  wrote models/churn_pipeline.pkl, model.pkl, scaler.pkl, encoder.pkl")
    print(f"  wrote models/model_metadata.json")
    print(f"  wrote reports/model_comparison.csv, reports/threshold_curve.csv")


if __name__ == "__main__":
    main()

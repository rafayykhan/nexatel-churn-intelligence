"""
Phase 2 — Exploratory Data Analysis.

Reads from the database (never the raw CSV — the SQL layer built in
Phase 1 is the source of truth), produces the figure set in
reports/figures/, and writes reports/eda_stats.json for the README and
the frontend dashboard to consume.

Run:  python src/eda.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from config import DB_PATH, FIGURES_DIR, PALETTE, REPORTS_DIR  # noqa: E402
from features import ADDON_SERVICES, engineer_features  # noqa: E402

V, C, NAVY, ROSE, GREEN = (PALETTE["violet"], PALETTE["cyan"], PALETTE["navy"],
                           PALETTE["rose"], PALETTE["green"])
CHURN_COLORS = [C, ROSE]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#0F172A",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.7,
    "font.size": 10,
    "text.color": "#0F172A",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})


def load() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM v_customer_360", conn)
    print(f"loaded {len(df):,} rows x {df.shape[1]} cols from v_customer_360")
    return df


def save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.name}")


# ---------------------------------------------------------------------
# 1. Target balance + univariate distributions
# ---------------------------------------------------------------------
def fig_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    counts = df["churn"].value_counts()
    axes[0].pie(counts, labels=[f"Retained\n{counts['No']:,}", f"Churned\n{counts['Yes']:,}"],
                colors=[C, ROSE], autopct="%1.1f%%", startangle=90,
                wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Churn balance (26.5% positive class)")

    for ax, col, title in zip(
        axes[1:], ["tenure", "monthly_charges", "total_charges"],
        ["Tenure (months)", "Monthly charges ($)", "Total charges ($)"],
    ):
        for churn, colour in zip(["No", "Yes"], CHURN_COLORS):
            sub = df.loc[df.churn == churn, col].dropna()
            ax.hist(sub, bins=30, alpha=0.65, color=colour, label=f"Churn={churn}")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Univariate distributions by churn outcome", fontsize=14,
                 fontweight="bold", y=1.03)
    save(fig, "01_distributions.png")


# ---------------------------------------------------------------------
# 2. Churn rate by every categorical driver
# ---------------------------------------------------------------------
def fig_categorical(df: pd.DataFrame) -> dict:
    cols = ["contract", "internet_service", "payment_method", "tech_support",
            "online_security", "paperless_billing", "dependents", "partner",
            "senior_citizen"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    overall = df.churn_flag.mean() * 100
    # The same rates feed the figure and the live dashboard, so the bars an
    # agent sees in the browser can never drift from the plotted chart.
    by_category: dict[str, dict[str, float]] = {}

    for ax, col in zip(axes.ravel(), cols):
        rate = (df.groupby(col)["churn_flag"].mean() * 100).sort_values(ascending=False)
        by_category[col] = {str(k): round(float(v), 2) for k, v in rate.items()}
        colours = [ROSE if v > overall else GREEN for v in rate.values]
        bars = ax.bar([str(i) for i in rate.index], rate.values, color=colours,
                      edgecolor="white", linewidth=1.2)
        ax.axhline(overall, color=V, linestyle="--", linewidth=1.4)
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("churn %")
        ax.set_ylim(0, max(rate.values) * 1.28)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for b, v in zip(bars, rate.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                    ha="center", fontsize=8, fontweight="bold")

    fig.suptitle(f"Churn rate by category  (dashed line = {overall:.1f}% company average)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "02_churn_by_category.png")
    return {"churn_by_category": by_category}


# ---------------------------------------------------------------------
# 3. Correlation / multicollinearity
# ---------------------------------------------------------------------
def fig_correlation(df: pd.DataFrame) -> dict:
    feats = engineer_features(df)
    num = feats.select_dtypes(include=[np.number]).copy()
    num["churn_flag"] = df["churn_flag"].values
    corr = num.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation matrix — engineered numeric features", pad=14)
    save(fig, "03_correlation_heatmap.png")

    target_corr = corr["churn_flag"].drop("churn_flag").sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colours = [ROSE if v > 0 else C for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colours, edgecolor="white")
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_title("Correlation with churn")
    ax.set_xlabel("Pearson r")
    save(fig, "04_target_correlation.png")

    # multicollinearity flags: |r| > 0.8 between predictors
    pairs = []
    cols = [c for c in corr.columns if c != "churn_flag"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.8:
                pairs.append({"a": a, "b": b, "r": round(float(r), 3)})
    return {"multicollinear_pairs": pairs,
            "target_correlation": {k: round(float(v), 3) for k, v in target_corr.items()}}


# ---------------------------------------------------------------------
# 4. Segment analysis — contract x tenure bucket
# ---------------------------------------------------------------------
def fig_segments(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["tenure_group"] = pd.cut(d.tenure, [-0.1, 12, 24, 48, np.inf],
                               labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    pivot = d.pivot_table(index="contract", columns="tenure_group",
                          values="churn_flag", aggfunc="mean") * 100
    pivot = pivot[["0-12", "13-24", "25-48", "49+"]]
    sizes = d.pivot_table(index="contract", columns="tenure_group",
                          values="churn_flag", aggfunc="size")[pivot.columns]

    labels = (pivot.round(1).astype(str) + "%\nn=" + sizes.astype(str))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(pivot, annot=labels.values, fmt="", cmap="Reds", linewidths=2,
                linecolor="white", cbar_kws={"label": "churn %"}, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Churn rate: contract type x tenure bucket")
    ax.set_xlabel("tenure group (months)")
    ax.set_ylabel("")
    save(fig, "05_segment_heatmap.png")

    worst = pivot.stack().idxmax()
    return {"worst_segment": {"contract": worst[0], "tenure_group": worst[1],
                              "churn_pct": round(float(pivot.stack().max()), 1),
                              "customers": int(sizes.loc[worst[0], worst[1]])}}


# ---------------------------------------------------------------------
# 5. Product depth and revenue at risk
# ---------------------------------------------------------------------
def fig_services_and_revenue(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["total_services"] = sum((d[c] == "Yes").astype(int) for c in ADDON_SERVICES)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    rate = d.groupby("total_services")["churn_flag"].mean() * 100
    n = d.groupby("total_services").size()
    axes[0].plot(rate.index, rate.values, marker="o", color=V, linewidth=2.5,
                 markersize=9, markerfacecolor="white", markeredgewidth=2.5)
    for x, y, cnt in zip(rate.index, rate.values, n.values):
        axes[0].annotate(f"{y:.1f}%\nn={cnt}", (x, y), textcoords="offset points",
                         xytext=(0, 11), ha="center", fontsize=8)
    axes[0].set_title("Churn vs. number of add-on services")
    axes[0].set_xlabel("add-on services subscribed")
    axes[0].set_ylabel("churn %")
    axes[0].set_ylim(0, rate.max() * 1.35)

    mrr = (d[d.churn_flag == 1].groupby("contract")["monthly_charges"].sum()
           .sort_values(ascending=False))
    bars = axes[1].bar(mrr.index, mrr.values, color=[ROSE, PALETTE["amber"], C],
                       edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, mrr.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 2000, f"${v:,.0f}",
                     ha="center", fontweight="bold", fontsize=9)
    axes[1].set_title("Monthly recurring revenue lost, by contract type")
    axes[1].set_ylabel("MRR at risk ($)")
    axes[1].set_ylim(0, mrr.max() * 1.18)

    fig.tight_layout()
    save(fig, "06_services_and_revenue.png")

    return {"churn_by_service_count": {int(k): round(float(v), 2) for k, v in rate.items()}}


# ---------------------------------------------------------------------
def data_quality(df: pd.DataFrame) -> dict:
    raw_total = pd.to_numeric(df["total_charges"], errors="coerce")
    return {
        "rows": int(len(df)),
        "duplicate_customer_ids": int(df.customer_id.duplicated().sum()),
        "missing_total_charges": int(raw_total.isna().sum()),
        "missing_all_tenure_zero": bool((df.loc[raw_total.isna(), "tenure"] == 0).all()),
        "zero_tenure_customers": int((df.tenure == 0).sum()),
        "negative_or_zero_charges": int((df.monthly_charges <= 0).sum()),
    }


def main() -> None:
    df = load()

    stats: dict = {"data_quality": data_quality(df)}
    stats["overall"] = {
        "customers": int(len(df)),
        "churned": int(df.churn_flag.sum()),
        "churn_rate_pct": round(float(df.churn_flag.mean() * 100), 2),
        "mrr_total": round(float(df.monthly_charges.sum()), 2),
        "mrr_at_risk": round(float(df.loc[df.churn_flag == 1, "monthly_charges"].sum()), 2),
        "annualised_at_risk": round(
            float(df.loc[df.churn_flag == 1, "monthly_charges"].sum() * 12), 2),
        "avg_tenure_churned": round(float(df.loc[df.churn_flag == 1, "tenure"].mean()), 1),
        "avg_tenure_retained": round(float(df.loc[df.churn_flag == 0, "tenure"].mean()), 1),
    }

    print("\nbuilding figures")
    fig_overview(df)
    stats.update(fig_categorical(df))
    stats.update(fig_correlation(df))
    stats.update(fig_segments(df))
    stats.update(fig_services_and_revenue(df))

    out = REPORTS_DIR / "eda_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(stats["overall"], indent=2))
    if stats["multicollinear_pairs"]:
        print("\nmulticollinearity (|r| > 0.8):")
        for p in stats["multicollinear_pairs"]:
            print(f"  {p['a']} <-> {p['b']}: r={p['r']}")


if __name__ == "__main__":
    main()

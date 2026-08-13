# NexaTel — Churn Prediction & Retention Intelligence

End-to-end data science project: a normalised SQL database, exploratory analysis,
feature engineering, model selection across five algorithms, SHAP explanations, and a
deployed web tool a retention agent can use without any data science background.

**Live demo:** _add your URL_ · **Repo:** _add your URL_

<!-- Screenshot: reports/figures/ has every chart referenced below -->

---

## The problem

A regional telecom is losing **26.5% of its subscriber base** — 1,869 of 7,043
customers, worth **$139,131 in monthly recurring revenue** (**$1.67M/year**). The
retention team only finds out a customer is unhappy *after* the cancellation goes
through, so offers are sent reactively: budget goes to customers who were never
leaving, while genuinely at-risk accounts get nothing.

This project predicts who is about to leave, explains why in plain language, and
recommends what to do about it.

## Results

| | |
|---|---:|
| **Recall (churn class)** | **89.6%** |
| **ROC-AUC** | **0.845** |
| Precision | 44.7% |
| False negatives cut | 81 → **39** |
| Revenue at risk identified | **$1.67M/year** |
| Modelled annual value (test set) | $63,691 |

Final model: **tuned Random Forest** (500 trees, depth 8, min_samples_leaf 10),
selected on cross-validated ROC-AUC across five candidates.

### The decision that matters most

The threshold is **0.31, not the default 0.50**.

A missed churner costs ~$893 in annual revenue. A wasted retention offer costs ~$35.
That is roughly **25:1**, so the model should deliberately over-flag. The threshold was
chosen where expected profit peaks — computed on **cross-validated training
predictions**, never on test, because selecting a threshold on held-out data makes the
reported score optimistic.

| Metric | t = 0.50 | t = 0.31 |
|---|---:|---:|
| Accuracy | 0.764 | 0.678 |
| Precision | 0.539 | 0.447 |
| **Recall** | 0.783 | **0.896** |
| False negatives | 81 | **39** |

Accuracy *falls* by 8.6 points. That is the correct trade: a model predicting "nobody
churns" scores 73.5% accuracy and identifies zero at-risk customers.

## Key findings

- **Contract type is the biggest lever.** Month-to-month churns at **42.7%**, two-year
  at **2.8%** — a 15x spread. Month-to-month holds 87% of all revenue at risk.
- **The worst pocket:** new (<6 months) + month-to-month + no tech support churns at
  **66.7%** across 904 customers. Two in three leave.
- **Churn is an onboarding problem.** 47.4% in months 0–12, 9.5% after four years.
- **Month-to-month fiber alone is 72% of revenue at risk** — $100,482 of $139,131.
- **Churn is not a price problem.** It rises with the bill up to $105, then *falls*.
  The most expensive customers are the most loyal — long-tenured, many add-ons. The
  intervention is bundle-and-contract, not discount.
- **SMOTE made things worse**, cutting recall 17 points. Almost certainly because the
  features are overwhelmingly categorical and interpolating between binary neighbours
  produces customers who cannot exist. Class weighting won.

## Tech stack

**Data** SQLite (3NF) · pandas · numpy
**ML** scikit-learn · XGBoost · imbalanced-learn · SHAP
**Backend** FastAPI · Pydantic · uvicorn
**Frontend** vanilla HTML/CSS/JS (no build step)
**Ops** Docker · Render · pytest

## Quickstart

```bash
git clone <your-repo> && cd nexatel-churn
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/load_to_db.py        # CSV -> normalised database
python src/run_sql_report.py    # 15 business queries -> reports/sql_results.md
python src/eda.py               # figures + eda_stats.json
python src/preprocess.py        # stratified split, fitted transformers
python src/train.py             # 5 models, tuning, threshold selection (~6 min)
python src/explain.py           # SHAP global + per-customer

cd backend && uvicorn app.main:app --reload --port 8000
# tool: http://localhost:8000/app/   ·   API docs: http://localhost:8000/docs
```

```bash
python -m pytest tests/ -q      # 26 passing
```

## Repository layout

```
sql/            schema.sql (3NF) + queries.sql (15 commented business queries)
src/            config · features · load_to_db · run_sql_report · eda
                preprocess · train · explain
notebooks/      01_eda · 02_feature_engineering · 03_modeling · 04_explainability
                (executed, with real outputs)
backend/        FastAPI app — schemas, service layer, routes
frontend/       agent tool: scoring form + insights dashboard
models/         churn_pipeline.pkl, model.pkl, scaler.pkl, encoder.pkl, metadata
reports/        figures, SQL results, model comparison, SHAP importance
docs/           problem statement, SQL findings, EDA insights, feature
                justification, model report, deployment, case study, resume
tests/          26 tests — feature invariants, API contracts, edge cases
```

## Documentation

| Phase | Document |
|---|---|
| 0 — Business understanding | [Problem statement](docs/00_problem_statement.md) |
| 1 — Database & SQL | [SQL findings](docs/01_sql_findings.md) · [results](reports/sql_results.md) |
| 2 — EDA | [Insights summary](docs/02_eda_insights.md) · [notebook](notebooks/01_eda.ipynb) |
| 3 — Feature engineering | [Justification](docs/03_feature_justification.md) · [notebook](notebooks/02_feature_engineering.ipynb) |
| 4–5 — Modelling | [Model report](docs/05_model_report.md) · [notebook](notebooks/03_modeling.ipynb) |
| 6 — Explainability | [notebook](notebooks/04_explainability.ipynb) |
| 8 — Deployment | [Deployment guide](docs/08_deployment.md) |
| 9 — Packaging | [Case study](docs/case_study.md) · [resume bullets](docs/resume_bullets.md) |

## Engineering notes

**One feature module, two consumers.** `src/features.py` is imported by both the
training pipeline and the live API. Reimplementing feature logic inside a backend is
the most common way a model quietly degrades after deployment; here there is exactly
one definition.

**Leakage is structural, not just checked.** The churn label lives in its own database
table, so reaching it requires an explicit JOIN that is visible in code review. No
target encoding, no churn-rate-by-segment features. Asserted in the test suite.

**Split before anything else.** The scaler and encoder are fitted on the training fold
only. SMOTE, where trialled, was applied inside each CV fold — resampling before the
split leaks synthetic neighbours of test rows into training.

**Explanations are deduplicated by concept.** `contract_ord` and
`contract_Month-to-month` encode one fact; an agent sees it once, phrased as a
sentence, not three times as raw column names.

## API

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'content-type: application/json' \
  -d '{"tenure":2,"monthly_charges":94.4,"contract":"Month-to-month",
       "internet_service":"Fiber optic","tech_support":"No",
       "payment_method":"Electronic check"}'
```

```json
{
  "churn_probability": 0.8836,
  "risk_level": "High",
  "risk_score": 88,
  "flagged_for_outreach": true,
  "revenue_at_risk_annual": 1132.8,
  "risk_factors": [
    {"label": "month-to-month contract", "impact": 0.0412},
    {"label": "months as a customer", "impact": 0.0326, "value": 2.0},
    {"label": "new + month-to-month + no tech support", "impact": 0.0233}
  ],
  "recommended_action": "Offer a 12-month contract with the first two months discounted..."
}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + loaded model |
| `POST /api/predict` | Score one customer with explanations |
| `POST /api/predict/batch` | Score up to 500 (nightly-run entry point) |
| `GET /api/model-info` | Metrics, params, threshold, economics |
| `GET /api/stats` | EDA figures + SHAP importance for the dashboard |
| `GET /docs` | Interactive Swagger UI |

## Limitations

Stated plainly, because knowing where a model is soft matters more than the headline
number:

- **The ~$22.6M full-book projection assumes a 30% intervention success rate** — an
  industry-typical figure, **not measured**. Every dollar figure scales linearly off
  it. Replace it with a number from a holdout retention experiment before budgeting.
- **This is a cross-sectional snapshot.** The model learns *who resembles a churner*,
  not *when* someone will leave. No time-to-churn estimate is possible.
- **Performance is capped by the features, not the algorithm.** Five very different
  model families landed within 0.03 AUC. The dataset has no support tickets, outage
  history, competitor pricing, or usage trend — typically the strongest churn signals
  in telecom.
- **Precision is 44.7% by design.** Roughly half of flagged customers were not going to
  leave. At $35 an offer against $893 of revenue, that is the intended trade, but it
  means the flag is a prioritised call list, not a verdict.

## Data

[IBM / Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 records, 21 fields. Used here as the "company database export".

## License

MIT

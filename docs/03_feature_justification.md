# Phase 3 — Feature Engineering & Justification

All features are defined once, in [`src/features.py`](../src/features.py), and
imported by both the training pipeline and the live API. Reimplementing feature logic
inside a backend is the most common cause of train/serve skew, so there is exactly one
definition and both paths call it.

**30 engineered columns → 54 model-matrix columns after encoding.**

## Data quality fixes applied first

| Issue found in EDA | Handling | Why not the alternative |
|---|---|---|
| `TotalCharges` stored as text with 11 blanks | Coerced to numeric, blanks → `0.0` | All 11 have `tenure = 0` — signed up, never billed. Mean imputation would invent ~$2,280 of billing history and make a brand-new customer look loyal to the model. |
| `tenure = 0` breaks the spend ratio | Denominator clipped to ≥ 1; ratio then overwritten with `monthly_charges` | Their historical average is undefined; the best available estimate of what they spend is their current plan price. |
| `'No internet service'` vs `'No'` | Kept distinct in the one-hot encoding | The first means the add-on was never purchasable; the second means it was offered and declined. Collapsing them destroys a real behavioural signal. |

## Engineered features

| # | Feature | Type | Justification | EDA support |
|---|---|---|---|---|
| 1 | `total_services` | count 0–6 | Count of optional add-ons held. Each additional product is a switching cost that makes leaving harder. | Churn falls from 45.8% at one add-on to 5.3% at six |
| 2 | `protection_services` | count 0–4 | Security/backup/device-protection/tech-support only, excluding streaming. Support-shaped products behave differently from entertainment ones and deserve their own signal. | Tech support alone: 41.6% vs 15.2% churn |
| 3 | `avg_monthly_spend_ratio` | float | `total_charges / tenure` — historical average spend. Separates a customer who has always paid $90 from one recently moved up to $90. | Churners average $74.44/mo vs $61.27 for retained |
| 4 | `charge_trend_delta` | float | `monthly_charges − avg_monthly_spend_ratio`. Positive means the customer is paying more now than they historically have: a recent upsell or price rise, which is what triggers a cancellation call. | Derived; no direct univariate equivalent |
| 5 | `contract_ord` | ordinal 0–2 | Contract length is genuinely ordered (commitment increases). One-hot alone would scatter the strongest effect in the data across three unordered dummies and discard the ordering. | 42.7% → 11.3% → 2.8% is monotone |
| 6 | `tenure_group` | categorical | Lifecycle buckets 0–12, 13–24, 25–48, 49+. | 47.4% → 28.7% → 20.4% → 9.5% |
| 7 | `tenure_group_ord` | ordinal 0–3 | Same buckets as an ordinal. Churn decays steeply rather than linearly with tenure; bucketing lets the linear baseline capture that shape. | Same as above |
| 8 | `new_customer_risk_flag` | binary | `tenure < 6` AND month-to-month AND no tech support. Trees can in principle discover this interaction alone; handing it over explicitly helps the linear baseline and makes the SHAP output legible to an agent. | This exact intersection churns at **66.7%** across 904 customers |
| 9 | `manual_payment_flag` | binary | Electronic or mailed check. A manual payment is a monthly re-decision to keep paying us. | 45.3% (e-check) vs 15.2% (auto card) |
| 10 | `no_protection_flag` | binary | Internet subscriber holding zero protection add-ons — unprotected and unsupported, with no safety net on their first bad experience. | Composite of the tech support / online security gaps |
| 11 | `is_fiber` | binary | Fiber is a distinct risk profile, not just "internet". | 41.9% churn at $91.50 avg vs 19.0% for DSL |

Raw demographic, service and billing columns are retained alongside these and one-hot
encoded, so nothing is lost by adding derived features.

## Encoding strategy

| Approach | Applied to | Reason |
|---|---|---|
| **Ordinal** | `contract`, `tenure_group` | Genuine ordering that carries signal |
| **One-hot** | All other nominal categoricals | No natural order; imposing one would be a false constraint |
| `drop='if_binary'` | Yes/No columns | Avoids a redundant column pair |
| `handle_unknown='ignore'` | All categoricals | If the retention team ever submits a payment method that did not exist at training time, the API returns a slightly less informed score instead of a 500 |
| **StandardScaler** | All numerics | Required by Logistic Regression, SVM and KNN; harmless for trees |

## Multicollinearity — found, and knowingly accepted

The correlation analysis flagged four predictor pairs above \|r\| = 0.8:

| Pair | r | Decision |
|---|---:|---|
| `monthly_charges` ↔ `avg_monthly_spend_ratio` | 0.996 | Kept |
| `tenure` ↔ `tenure_group_ord` | 0.961 | Kept |
| `total_services` ↔ `protection_services` | 0.913 | Kept |
| `tenure` ↔ `total_charges` | 0.826 | Kept |

These are kept deliberately, not overlooked:

- **The final model is a Random Forest.** Tree splits are threshold comparisons;
  correlated inputs affect which of two near-identical features gets chosen at a
  split, not predictive validity.
- **The linear baseline uses L2 regularisation** (tuned to `C = 0.011`, i.e. strong
  shrinkage), which is the standard remedy for inflated coefficient variance under
  collinearity.
- **The pairs are informative, not redundant.** `charge_trend_delta` is only meaningful
  *because* `monthly_charges` and `avg_monthly_spend_ratio` are both present — their
  difference is the signal.

The trade-off: individual Random Forest feature importances are diluted across
correlated pairs. That is why the explanation layer deduplicates by business concept
before showing an agent a reason, rather than showing raw column importances.

## Leakage check

Every feature is computable from information that exists **before** a customer
cancels: demographics, contract terms, subscribed products, and billing to date.

- Nothing reads `churn` or `churn_flag`.
- No target encoding, and no churn-rate-by-segment feature — either would leak the
  label through an aggregate and produce a model that validates beautifully and fails
  on the first genuinely unseen customer.
- `total_charges` is historical billing, not a post-cancellation settlement figure.
- The label lives in its own database table, so reaching it requires an explicit JOIN
  that would be visible in review.

Asserted in [`tests/test_features.py::test_no_feature_encodes_the_target`](../tests/test_features.py).

# Phase 5 — Model Report

Produced by [`src/train.py`](../src/train.py). Full run log: [`reports/train_log.txt`](../reports/train_log.txt).

## Setup

| | |
|---|---|
| Split | 80/20 stratified — 5,634 train / 1,409 test |
| Validation | 5-fold stratified cross-validation on train |
| Class ratio | 2.77 : 1 (26.54% positive) |
| Imbalance handling | `class_weight='balanced'` / `scale_pos_weight` |
| Selection metric | ROC-AUC (threshold-independent), recall as the operating constraint |

Preprocessing is fitted **on the training fold only** and applied to test. The scaler
never sees test data — the single most commonly asked-about mistake in interviews, and
the one that most reliably inflates a reported score without improving the model.

## Candidate comparison

Untuned, evaluated at the default 0.50 threshold:

| Model | CV ROC-AUC | Test ROC-AUC | Recall | Precision | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8463 | 0.8436 | 0.775 | 0.511 | 0.616 | 0.651 |
| Random Forest | 0.8419 | 0.8368 | 0.666 | 0.565 | 0.611 | 0.644 |
| XGBoost | 0.8395 | 0.8372 | 0.759 | 0.528 | 0.623 | 0.650 |
| SVM (RBF) | 0.8245 | 0.8246 | 0.639 | 0.573 | 0.604 | 0.606 |
| KNN | 0.8215 | 0.8142 | 0.513 | 0.602 | 0.554 | 0.590 |

**Logistic Regression is a genuinely strong baseline here** — churn in the engineered
feature space is close to linearly separable. That is worth stating plainly rather
than burying: the eventual winner beats it by a small margin, and if this had to ship
on a CPU-constrained service, the linear model would be a defensible choice.

CV and test scores agree to within ~0.003 AUC across every model, which indicates the
split is representative and nothing is overfitting the training fold.

## Imbalance strategy: SMOTE vs class weighting

Tested head-to-head on XGBoost, with SMOTE applied **inside each CV fold only** —
resampling before the split would leak synthetic neighbours of test rows into training.

| Strategy | CV ROC-AUC | CV Recall | CV F1 |
|---|---:|---:|---:|
| `scale_pos_weight` | **0.8395** | **0.745** | **0.626** |
| SMOTE | 0.8375 | 0.578 | 0.597 |

SMOTE **hurt recall by 17 points** — the opposite of what it is usually reached for.
The likely reason: the minority class here is not sparse or clustered in a way that
benefits from synthetic interpolation. Nearly all features are categorical or binary,
and SMOTE's linear interpolation between neighbours produces fractional values for
attributes that only exist as 0 or 1. Class weighting achieves the same rebalancing
without inventing customers who cannot exist, and keeps probability outputs closer to
calibrated — which matters because the threshold is chosen on those probabilities.

**Decision: class weighting.**

## Tuning

`RandomizedSearchCV`, 15 iterations × 5 folds, scored on ROC-AUC, applied to the two
strongest candidates by CV AUC.

| Model | CV AUC before | CV AUC after | Best parameters |
|---|---:|---:|---|
| Logistic Regression | 0.8463 | 0.8467 | `C=0.0113`, `penalty=l2`, `solver=liblinear` |
| Random Forest | 0.8419 | **0.8479** | `n_estimators=500`, `max_depth=8`, `min_samples_leaf=10`, `max_features='sqrt'` |

Random Forest gained the most from tuning. The winning configuration is a *constrained*
forest — depth capped at 8, minimum 10 samples per leaf — which is the tuner correcting
overfitting in the default configuration rather than adding capacity.

## Final model: tuned Random Forest

Selected on cross-validated ROC-AUC (0.8479), the highest of any candidate. The margin
over tuned Logistic Regression (0.8467) is 0.0012 — small enough to acknowledge
honestly. Random Forest is chosen because it also carries the better precision/recall
balance across the threshold sweep and is unaffected by the multicollinearity
documented in the feature notes.

## Choosing the operating threshold

The default 0.50 threshold assumes a false positive and a false negative cost the same
amount. Here they do not, by roughly 25:1.

**Cost model:**

| Parameter | Value | Source |
|---|---|---|
| Retention offer cost | $35 | Assumption, stated in `src/config.py` |
| Intervention success rate | 30% | Industry-typical assumption — **not measured** |
| Revenue horizon | 12 months | Conservative; uses each customer's own MRR |

Expected value at a given threshold = (revenue saved on true positives × save rate)
− (offer cost × everyone flagged).

The threshold is selected on **cross-validated predictions over the training set**.
Selecting it on test would be tuning a hyperparameter against held-out data and would
make the reported test recall optimistic — the same class of error as fitting a scaler
on test.

**Selected threshold: 0.31.**

| Metric | t = 0.50 | t = 0.31 | Change |
|---|---:|---:|---:|
| Accuracy | 0.764 | 0.678 | −8.6 pts |
| Precision | 0.539 | 0.447 | −9.2 pts |
| **Recall** | 0.783 | **0.896** | **+11.3 pts** |
| F1 | 0.638 | 0.596 | −4.2 pts |
| ROC-AUC | 0.845 | 0.845 | unchanged |

**Confusion matrix on the 1,409-customer test set:**

| | t = 0.50 | t = 0.31 |
|---|---|---|
| True negatives | 784 | 620 |
| False positives | 251 | 415 |
| False negatives | **81** | **39** |
| True positives | 293 | 335 |

Moving the threshold catches **42 additional churners** at the cost of **164 additional
retention offers**. At $35 per offer and roughly $893 in annual revenue per churner,
that trade is strongly positive. Note that F1 *falls* — the model was deliberately not
optimised for F1, because F1 weights precision and recall equally and this business
problem does not.

## Modelled financial impact

| | Value |
|---|---:|
| Annual value on the 1,409-customer test set, t=0.50 | $59,304 |
| Annual value on the same set, t=0.31 | **$63,691** |
| Gain from threshold selection alone | +$4,386 (+7.4%) |
| Projected across a 500,000-subscriber book | ~$22.6M/year |

**Read the projection with care.** It is a linear scale-up of test-set performance and
assumes the full book resembles the sample and that the 30% save rate holds. It is
directionally useful for prioritisation and should not be entered into a budget until
the save rate is measured in a holdout retention experiment.

## What would improve this model

Performance is capped by what the dataset contains, not by algorithm choice — five
very different model families landed within 0.03 AUC of each other, which is the
signature of a feature ceiling rather than a modelling one.

The highest-value additions would be **event data**: support ticket volume and
sentiment, outage history on the customer's line, competitor pricing in their area,
and usage trend (is data consumption falling?). A falling-usage signal in particular is
typically among the strongest leading indicators of telecom churn, and nothing in this
snapshot captures it.

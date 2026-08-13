# Resume Bullets & Interview Prep

Every number below comes from this repo and is reproducible. Do not inflate them —
the ability to defend a modest number is worth more in an interview than a big one you
cannot source.

---

## Resume bullets

Pick two or three. The first is the strongest general-purpose one.

**Full-stack ML (lead with this):**

> Built and deployed an end-to-end customer churn prediction system on 7,043 telecom
> subscriber records — 3NF SQL database, feature engineering, model selection across 5
> algorithms, and a SHAP-explained FastAPI web tool — achieving **89.6% recall** and
> **0.845 ROC-AUC** on held-out data against **$1.67M** in identified annual revenue at
> risk.

**Business-impact framing:**

> Identified that new month-to-month customers without tech support churn at **66.7%**
> (vs 26.5% baseline) and set the model's decision threshold on an expected-profit
> curve rather than the 0.5 default, cutting false negatives **from 81 to 39** on the
> test set and improving modelled annual retained revenue by **7.4%**.

**Engineering rigour:**

> Engineered a leakage-free ML pipeline with a single shared feature module imported by
> both training and the production API to eliminate train/serve skew; validated with
> **26 automated tests** covering feature invariants, API contracts and edge cases, and
> shipped as a Dockerised single-service deployment.

**Data engineering angle:**

> Normalised a flat 21-column extract into a **3NF SQLite schema** and authored **15
> documented business SQL queries** answering stakeholder questions, isolating the
> target variable in its own table so accidental label leakage is visible in code
> review.

**One-line version** (for a dense resume):

> Churn prediction system — SQL → EDA → SHAP-explained Random Forest (89.6% recall,
> 0.845 AUC) → deployed FastAPI tool. $1.67M annual revenue at risk identified.

---

## LinkedIn / portfolio blurb

> I built a churn prediction system for a telecom scenario and shipped it end to end —
> from a normalised SQL database through to a live tool a retention agent can use
> without any data science background.
>
> The most interesting decision wasn't the model. It was the **threshold**. Everyone
> defaults to 0.5, but a missed churner costs ~$893 in annual revenue while a wasted
> retention offer costs ~$35 — roughly 25:1. Setting the threshold where expected
> profit peaks (0.31) caught 42 more churners for 164 extra offers.
>
> It *lowered* accuracy from 76% to 68%. That's the right trade, and being able to
> explain why is the actual skill.
>
> The other surprise: SMOTE made things worse. It cut recall by 17 points, most likely
> because the features are overwhelmingly categorical and interpolating between binary
> neighbours produces customers who can't exist. Class weighting won.

---

## Interview questions you should expect

**"Why did you pick recall over accuracy?"**
Churn is 26.5% of the book, so predicting "nobody churns" scores 73.5% accuracy and
catches zero at-risk customers. The errors aren't symmetric either — a false negative
loses ~$893 of annual revenue, a false positive costs a ~$35 offer. About 25:1. Recall
is what the business actually pays for.

**"How did you choose 0.31?"**
I built an expected-value curve: revenue saved on true positives × assumed save rate,
minus offer cost across everyone flagged. The peak is at 0.31. Critically, I selected
it on **cross-validated training predictions**, not on test — picking a threshold on
held-out data is tuning a hyperparameter against your own evaluation set, the same
class of error as fitting a scaler on test.

**"How do you know there's no leakage?"**
Three layers. Structurally, the label lives in a separate database table, so reaching
it needs an explicit JOIN. In the feature module, every feature is computable before a
cancellation — no target encoding, no churn-rate-by-segment aggregates. And there's an
assertion in the test suite that the target never appears in the feature matrix.

**"Random Forest only beat Logistic Regression by 0.0012 AUC. Was it worth it?"**
Honestly, marginally. I'd say that in the room. Five very different model families
landed within 0.03 AUC of each other, which tells me I hit a *feature* ceiling, not a
modelling one. If this had to run on constrained hardware I'd ship the logistic model
and lose almost nothing. The bigger win would be event data — support tickets, outage
history, usage trend.

**"Why did SMOTE fail?"**
Nearly every feature is categorical or binary. SMOTE interpolates linearly between
minority neighbours, which produces fractional values for attributes that only exist as
0 or 1 — synthetic customers who can't exist. Class weighting rebalances without
inventing anyone, and keeps probabilities better calibrated, which matters because my
threshold is chosen on those probabilities.

**"You kept features correlated at r=0.996. Why?"**
Deliberately. The final model is a tree ensemble — splits are threshold comparisons, so
collinearity affects which of two near-identical features gets picked, not validity.
The linear baseline uses strong L2 (C=0.011), the standard remedy. And the pair is
informative: `charge_trend_delta` is only meaningful *because* both current and
historical spend are present — their difference is the signal.

**"What would you do differently?"**
Two things. First, I'd measure the intervention success rate instead of assuming 30% —
every dollar figure scales linearly off that assumption, and it's the softest number in
the project. Second, this is a cross-sectional snapshot, so the model learns *who
resembles a churner*, not *when* someone will leave. With timestamped event data I'd
frame it as survival analysis and get a time-to-churn estimate, which is far more
actionable for scheduling outreach.

**"Walk me through what happens when an agent scores a customer."**
The browser posts to `/api/predict`. The service loads artifacts once at startup, runs
the request through `engineer_features` — the exact same module training used, which is
how I avoid train/serve skew — then the fitted preprocessor and the Random Forest.
SHAP's TreeExplainer produces per-feature contributions, which I deduplicate by
business concept so the agent doesn't see "contract length" and "month-to-month
contract" as two separate reasons, and a rules playbook maps the top driver to a
specific offer.

---

## Numbers to have memorised

| | |
|---|---|
| Dataset | 7,043 subscribers, 21 raw fields |
| Churn rate | 26.54% (1,869 customers) |
| Revenue at risk | $139,131/month, $1.67M/year |
| Final model | Random Forest (500 trees, depth 8, min_leaf 10) |
| Test recall / AUC | 89.6% / 0.845 |
| Threshold | 0.31 (vs 0.50 default) |
| False negatives | 81 → 39 |
| Worst segment | New + month-to-month + no tech support = 66.7% |
| Biggest lever | Contract: 42.7% M2M vs 2.8% two-year |
| Feature count | 30 engineered → 54 encoded |
| Tests | 26 passing |

**The one caveat to volunteer before you're asked:** the ~$22.6M annual projection
assumes a 30% intervention success rate that is industry-typical, not measured.
Flagging your own soft number before an interviewer finds it reads as rigour. Being
caught defending it reads as the opposite.

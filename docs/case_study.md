# Predicting Customer Churn for a Regional Telecom

**A one-page case study — problem, approach, insight, result, impact.**

---

## Problem

A regional telecom with roughly 500,000 subscribers was losing **26.5% of its customer
base** every measurement period — about **$1.67 million in annual recurring revenue**.

The retention team's problem was not effort, it was timing. They only learned a
customer was unhappy after the cancellation went through. Retention offers were sent
reactively, which meant budget went to customers who were never going to leave, while
genuinely at-risk accounts got nothing.

They needed to know *who* was about to leave, *why*, and *what to do about it* —
delivered to people with no data science background.

## Approach

I built the whole path from raw data to a tool an agent can open in a browser.

**Database first.** The flat 7,043-row extract was normalised into a 3NF SQLite schema
— customers, accounts, services, and churn status as separate tables. The churn label
lives in its own table on purpose: reaching it requires an explicit JOIN, which makes
accidental target leakage visible in code review instead of invisible in a dataframe.
Fifteen commented business queries answer the stakeholders' actual questions.

**Analysis before modelling.** Exploratory analysis from the database, not the CSV,
established where losses concentrate and produced the feature hypotheses that followed.

**Features with reasons.** Eleven engineered features, each justified against a
specific finding — product depth, spend trend, lifecycle stage, and a composite risk
flag for the worst-performing segment. All defined in one module that both the training
pipeline and the production API import, so there is no possibility of train/serve skew.

**Five models, honestly compared.** Logistic Regression, Random Forest, XGBoost, SVM
and KNN under identical preprocessing, with 5-fold cross-validation. I also tested
SMOTE against class weighting head-to-head — SMOTE *hurt* recall by 17 points, so it
was dropped, and the reasoning is documented rather than hidden.

**Explanations, not just scores.** SHAP produces the top factors behind each individual
prediction, deduplicated by business concept and rewritten as sentences an agent can
say on a call, paired with a specific retention offer.

## Key insight

**Churn is not spread across the book — it sits in a small, addressable pocket.**

Customers who were **new (under six months), on a month-to-month contract, with no tech
support** churned at **66.7%**. Two out of every three left. That is 904 customers
carrying $40,837 in monthly revenue.

More broadly: month-to-month customers churn at 42.7% versus **2.8% on a two-year
contract** — a 15x difference, and the single largest lever in the business. Month-to-
month fiber customers alone account for **72% of all revenue at risk**.

The unexpected finding was about price. Churn rises with the monthly bill up to
$105 — and then *falls*. The most expensive customers are the most loyal, because they
are long-tenured and hold many add-on services. Churn is not a price problem, it is a
**commitment and product-depth problem**, and that changes the recommended intervention
completely: bundle and contract, don't discount.

## Result

A tuned Random Forest achieving **89.6% recall** and **0.845 ROC-AUC** on a held-out
test set of 1,409 customers.

The decision threshold was deliberately set to **0.31 rather than 0.50**, chosen where
expected profit peaks on cross-validated training predictions. Because a missed churner
costs ~$893 in annual revenue while a wasted retention offer costs ~$35, the economics
are roughly 25:1 in favour of over-flagging. That single choice cut false negatives
from 81 to 39 — **42 more churners caught** for 164 extra offers.

Notably, this *lowered* accuracy from 76% to 68%. Optimising for accuracy would have
been optimising for the wrong thing: a model predicting "nobody churns" scores 73.5%
accuracy and identifies zero at-risk customers.

The whole thing ships as a live web tool — an agent enters a customer's details and
gets a risk score, the top three drivers in plain language, and a specific recommended
offer.

## Business impact

| | |
|---|---:|
| Revenue identified as at risk | **$1.67M/year** |
| Churners caught by the model (recall) | **89.6%** |
| Modelled annual value, test set | **$63,691** |
| Projected across a 500k-subscriber book | **~$22.6M/year** |

The projection assumes a 30% intervention success rate — an industry-typical figure,
not a measured one. I have flagged it as the first thing that should be replaced with
a real number from a holdout retention experiment before anyone budgets against it.

**Tech:** Python · SQL (SQLite, 3NF) · pandas · scikit-learn · XGBoost · imbalanced-learn ·
SHAP · FastAPI · JavaScript · Docker · pytest

**Live demo:** _[add your deployed URL]_ · **Code:** _[add your GitHub URL]_

# Phase 2 — Insights Summary

**To:** VP of Customer Retention
**Re:** What the subscriber data says about who is leaving
**Basis:** All 7,043 subscriber records, queried from the analytics database

---

We are losing **26.5% of subscribers — 1,869 customers — worth $139,131 in monthly
recurring revenue, or about $1.67 million a year.** Churn is not spread evenly across
the book. It is concentrated in a small number of identifiable segments, which is good
news: it means retention spend can be targeted rather than broadcast.

## 1. Contract type is the single biggest lever we have

| Contract | Customers | Churn rate | MRR at risk |
|---|---:|---:|---:|
| Month-to-month | 3,875 | **42.7%** | $120,847 |
| One year | 1,473 | 11.3% | $14,119 |
| Two year | 1,695 | **2.8%** | $4,165 |

A customer on a two-year contract is **15 times less likely to leave** than one on
month-to-month. Month-to-month accounts hold **87% of all revenue at risk**. Anything
that moves customers onto a term contract is the highest-leverage action available.

## 2. We lose people in the first year, not later

| Tenure | Customers | Churn rate |
|---|---:|---:|
| 0–12 months | 2,186 | **47.4%** |
| 13–24 months | 1,024 | 28.7% |
| 25–48 months | 1,594 | 20.4% |
| 49+ months | 2,239 | **9.5%** |

Nearly half of first-year customers leave. This is an **onboarding problem**, not a
loyalty problem — retention effort spent on long-tenured customers is largely wasted,
because they were not going anywhere.

## 3. The worst pocket in the book

Customers who are **new (under 6 months), on month-to-month, with no tech support**
churn at **66.7%** — 603 of 904 customers, carrying $40,837 in monthly revenue.

Two out of three of these customers leave. This one segment is a well-defined,
addressable retention programme on its own.

## 4. Fiber is a problem product

| Internet service | Customers | Churn rate | Avg bill |
|---|---:|---:|---:|
| Fiber optic | 3,096 | **41.9%** | $91.50 |
| DSL | 2,421 | 19.0% | $58.10 |
| None | 1,526 | 7.4% | $21.08 |

Our premium product churns at more than twice the rate of the cheaper one. This is
worth investigating directly — the pattern is consistent with an expectation gap
(customers pay a premium and do not feel they receive one) rather than pure price
sensitivity, because the highest-priced band ($105+) actually churns *less* (21.3%)
than the $85–105 band (37.9%). The customers paying the very most are long-tenured
and loaded with add-ons.

**Month-to-month fiber alone is 72% of all revenue at risk** — $100,482 of the
$139,131 monthly total. If we work one segment, this is it.

## 5. How people pay predicts whether they stay

| Payment method | Churn rate |
|---|---:|
| Electronic check | **45.3%** |
| Mailed check | 19.1% |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.2% |

Manual payment means the customer makes an active decision to keep paying us every
single month. Automatic payment removes that decision point. The gap is 30 points.

## 6. Depth of relationship protects

| Add-on services held | Churn rate |
|---|---:|
| 1 | **45.8%** |
| 2 | 35.8% |
| 3 | 27.4% |
| 4 | 22.3% |
| 5 | 12.4% |
| 6 | **5.3%** |

Every additional product a customer holds makes leaving harder. Note that customers
with **zero** add-ons churn at only 21.4% — that group is mostly phone-only customers
on cheap plans who were never at much risk. The real danger zone is a customer with
**one** add-on: engaged enough to have internet, not embedded enough to stay.

Support-type add-ons matter more than entertainment ones. Internet customers without
tech support churn at **41.6%** versus **15.2%** with it.

## 7. Churners pay more and stay less

| | Churned | Retained |
|---|---:|---:|
| Average tenure | 18.0 months | 37.6 months |
| Average monthly bill | $74.44 | $61.27 |

We are not losing cheap customers. We are losing **premium customers, early**.

## Data quality note

The extract is clean: 7,043 rows, no duplicate customer IDs, no negative or zero
charges. Eleven customers have a blank total-billed figure — all eleven have zero
months of tenure, meaning they signed up but have not yet been billed. That is a real
zero, not missing data, and it is handled as such.

## What I would do with this

1. **Target the 904-customer new/month-to-month/no-support segment first.** Highest
   churn rate in the book, and every intervention has a clear lever attached.
2. **Push month-to-month fiber customers onto term contracts.** 72% of revenue at risk
   sits here.
3. **Bundle tech support into fiber plans by default.** The 26-point churn gap on tech
   support is the cheapest structural fix available.
4. **Move manual payers to autopay** with a small one-time credit.
5. **Concentrate onboarding effort in months 0–12.** That is where half the losses are.

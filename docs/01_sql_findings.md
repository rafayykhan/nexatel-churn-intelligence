# Phase 1 — SQL Findings Summary

Schema: [`sql/schema.sql`](../sql/schema.sql) · Queries: [`sql/queries.sql`](../sql/queries.sql)
Live output of every query: [`reports/sql_results.md`](../reports/sql_results.md)

Regenerate with:

```bash
python src/load_to_db.py      # CSV -> normalised database
python src/run_sql_report.py  # runs all 15 queries -> reports/sql_results.md
```

## Schema design

The source is one flat 21-column extract. A flat table repeats customer-level facts
next to service-level and billing-level facts, so a change to a payment method touches
rows that also carry demographics — the classic update anomaly.

Normalised to 3NF, grouped by the entity that owns each attribute:

| Table | Holds | Rows |
|---|---|---:|
| `customers` | demographics + tenure | 7,043 |
| `accounts` | contract, billing, charges | 7,043 |
| `services` | subscribed products and add-ons | 7,043 |
| `churn_status` | the label, isolated | 7,043 |

Two deliberate choices:

- **Churn lives in its own table.** Reaching the label requires an explicit JOIN, which
  makes accidental target leakage visible in review. It also mirrors reality — churn is
  usually recorded by a different system than billing.
- **`v_customer_360` is the analyst-facing view.** The tables are normalised for
  integrity; analysts and the EDA layer read the view so nobody hand-rolls the
  four-way join and gets it subtly wrong.

`total_charges` is nullable on purpose: brand-new customers have never been billed, and
storing a real `NULL` rather than the source's blank string makes that missingness
queryable.

## Integrity verification

| Check | Result |
|---|---|
| Rows loaded per table | 7,043 |
| Duplicate customer IDs | 0 |
| Orphan foreign keys | 0 |
| NULL `total_charges` | 11 |
| ...unexplained (tenure ≠ 0) | **0** |
| Non-positive monthly charges | 0 |
| Customers missing a services row | 0 |

Every anomaly in the extract is explained. The eleven NULLs are exactly the eleven
customers with `tenure = 0` — signed up, not yet billed.

## Findings

**Q01–Q02 — the headline.** 1,869 of 7,043 customers churned (**26.54%**), carrying
**$139,131 in monthly recurring revenue** — **$1,669,570 annualised**. The average
churner bills $74.44/month.

**Q03 — contract is the dominant lever.** Month-to-month churns at **42.71%**, one-year
at 11.27%, two-year at **2.83%**. A 15x spread, and month-to-month holds $120,847 of
the $139,131 at risk (87%).

**Q04 — fiber is the problem product.** Fiber optic churns at **41.89%** on a $91.50
average bill; DSL at 18.96% on $58.10; no internet at 7.40%.

**Q05 — churners are premium and early.** Average tenure 18.0 months vs 37.6 for
retained customers; average bill $74.44 vs $61.27. We lose expensive customers young.

**Q06 — manual payment predicts leaving.** Electronic check **45.29%**, mailed check
19.11%, bank transfer 16.71%, automatic credit card 15.24%. Manual payment forces a
monthly decision to keep paying.

**Q07 — the call list.** Month-to-month + electronic check is the worst segment at
**53.73%** churn across 1,850 customers ($77,316 MRR at risk) — the top segment by
both rate and value.

**Q08 — tech support is a retention product.** Among internet subscribers, **41.64%**
churn without tech support vs **15.17%** with it. Restricted to internet customers on
purpose: the add-on cannot be sold to anyone else, so including them would dilute the
comparison.

**Q09 — the worst pocket in the book.** New (<6 months) + month-to-month + no tech
support: **66.70%** churn, 603 of 904 customers, $40,837 MRR at risk. Two in three
leave.

**Q10 — depth protects, but not linearly.** Churn by add-on count: 0 → 21.41%,
1 → **45.76%**, 2 → 35.82%, 3 → 27.37%, 4 → 22.30%, 5 → 12.43%, 6 → **5.28%**. The
zero-add-on group is *not* the riskiest — those are mostly phone-only customers on
$32 plans who were never at much risk. The danger zone is **one** add-on: engaged
enough to have internet, not embedded enough to stay.

**Q11 — churn is an onboarding problem.** 0–12 months: **47.44%**. 13–24: 28.71%.
25–48: 20.39%. 49+: **9.51%**. Nearly half of first-year customers leave.

**Q12 — churn is not a price problem.** Churn rises with the bill up to $105
(10.86% → 25.60% → 32.73% → **37.90%**) and then *falls* to 21.29% in the $105+ band.
The most expensive customers are the most loyal — they average 58 months of tenure and
hold many add-ons. This reframes the intervention entirely: bundle and contract, don't
discount.

**Q13 — demographics matter less than behaviour.** The worst demographic cut (senior,
no partner, no dependents) churns at 49.20% — real, but weaker than contract type, and
not a targetable lever. Behaviour beats demography here.

**Q14 — where the money actually is.** Month-to-month + fiber: 2,128 customers,
54.61% churn, **$100,482 MRR at risk — 72.2% of all revenue at risk**. If retention
works exactly one segment, this is it.

**Q15 — data quality gate.** Run before any modelling; every count is explainable
(see the integrity table above).

## What the SQL layer established for later phases

1. Contract type, tenure, internet product, payment method and add-on depth are the
   real signals — this shaped the entire feature set.
2. The 66.7% segment from Q09 was encoded directly as `new_customer_risk_flag`.
3. The Q12 non-linearity in price justified bucketing rather than treating charges as
   a straight-line effect.
4. The 26.54% base rate is what makes accuracy a useless metric downstream.

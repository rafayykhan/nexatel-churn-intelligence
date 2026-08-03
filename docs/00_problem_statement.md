# Phase 0 — Problem Statement

## The problem, stated formally

NexaTel Communications is losing 26.5% of its subscriber base to voluntary
cancellation, and currently has no way to identify an at-risk customer before the
cancellation happens. Retention spend is therefore allocated reactively — offers
reach customers who were never going to leave, while genuinely at-risk accounts go
untouched. **This project builds a supervised binary classifier that assigns each
subscriber a cancellation probability from account, service and billing attributes
available today, together with a per-customer explanation a retention agent can act
on without any data science background.**

## Target variable

`Churn` — a binary label, `Yes` (1) or `No` (0), recorded per customer.

- Positive class: 1,869 of 7,043 customers (26.54%)
- Class ratio: 2.77 negative to 1 positive

## Which error costs more

A **false negative** — a customer the model says is safe, who cancels — costs NexaTel
the entire relationship. Churners bill $74.44/month on average, so a missed churner
is roughly **$893 of annual recurring revenue**, before any acquisition cost to
replace them.

A **false positive** — a retention offer sent to a customer who was staying anyway —
costs one intervention, roughly **$35**.

The asymmetry is about **25:1**. This drives every downstream decision:

- **Recall is the primary operating metric**, with ROC-AUC used for model selection
  because it is threshold-independent.
- **Accuracy is explicitly rejected.** Predicting "nobody churns" scores 73.5%
  accuracy and identifies zero at-risk customers.
- **The decision threshold is not 0.5.** It is set where expected profit peaks, which
  the analysis puts at **0.31** — deliberately over-flagging, because over-flagging is
  25x cheaper than under-flagging.

## Business questions the analysis must answer

1. What is the overall churn rate, and how many subscribers does it represent?
2. How much monthly and annual recurring revenue is attached to churned customers?
3. Does contract type affect churn, and by how much?
4. Is churn concentrated in a particular internet product?
5. How does the tenure of a churner compare to a customer who stays?
6. Which payment method correlates with leaving?
7. Are customers without tech support more likely to cancel?
8. Which contract × payment-method segments churn hardest?
9. Does subscribing to more add-on services make a customer stickier?
10. Where in the customer lifecycle does churn concentrate?
11. Is churn driven by price level, or by something else?
12. If retention could work only one segment, which returns the most revenue?

## Revenue at risk — the headline statistic

| Measure | Value |
|---|---|
| Churned customers | 1,869 |
| Average monthly charge of a churner | $74.44 |
| **Monthly recurring revenue lost** | **$139,131** |
| **Annualised** | **$1,669,570** |

Against NexaTel's stated ~500,000 subscriber book, the 7,043-record extract is a
sample; the modelled economics are scaled to the full book in the model report and
clearly labelled as a projection rather than a measured figure.

## Definition of success

| Dimension | Target |
|---|---|
| Statistical | Recall ≥ 0.80 on the churn class, ROC-AUC ≥ 0.83 on held-out data |
| Explanatory | Every prediction carries the top factors driving it, in plain language |
| Operational | A retention agent with no analytics background can score a customer unaided |
| Financial | A defensible dollar figure for revenue protected, with assumptions stated |

## Scope and honest limitations

This is a **cross-sectional snapshot**, not a time series. Each customer appears once,
with churn already recorded. Consequences worth stating up front:

- The model learns *who resembles a churner*, not *when* a given customer will leave.
  It cannot produce a time-to-churn estimate.
- There is no event history — no support tickets, no outage records, no complaint
  logs, no competitor pricing. Those are typically among the strongest churn signals
  in a real telecom, and their absence caps achievable performance.
- The 30% intervention success rate used in the cost model is an industry-typical
  assumption, not a measured NexaTel figure. Every dollar figure in this project
  scales linearly with it, and it should be replaced with a measured rate from a
  holdout retention experiment before anyone budgets against it.

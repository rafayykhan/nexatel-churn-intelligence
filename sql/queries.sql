-- =====================================================================
-- NexaTel Communications — Business Query Library
-- Dialect: SQLite 3 (runs on PostgreSQL with the noted substitutions)
--
-- Each query is preceded by the business question it answers and who
-- asked for it. Run them all with: python src/run_sql_report.py
--
-- Query IDs Q01..Q15 are referenced by docs/01_sql_findings.md.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Q01 | VP of Retention
-- "What is our overall churn rate, and how many subscribers is that?"
-- The headline number everything else is measured against.
-- ---------------------------------------------------------------------
SELECT
    COUNT(*)                                        AS total_customers,
    SUM(churn_flag)                                 AS churned_customers,
    ROUND(AVG(churn_flag) * 100, 2)                 AS churn_rate_pct
FROM churn_status;


-- ---------------------------------------------------------------------
-- Q02 | Finance
-- "What monthly recurring revenue walked out the door?"
-- Revenue at risk = MRR attached to customers flagged as churned.
-- ---------------------------------------------------------------------
SELECT
    COUNT(*)                                        AS churned_customers,
    ROUND(SUM(a.monthly_charges), 2)                AS monthly_revenue_lost,
    ROUND(SUM(a.monthly_charges) * 12, 2)           AS annualised_revenue_lost,
    ROUND(AVG(a.monthly_charges), 2)                AS avg_monthly_charge_of_churner
FROM accounts a
JOIN churn_status ch ON ch.customer_id = a.customer_id
WHERE ch.churn_flag = 1;


-- ---------------------------------------------------------------------
-- Q03 | VP of Retention
-- "Does contract type affect churn?"
-- The single strongest lever in the whole dataset.
-- ---------------------------------------------------------------------
SELECT
    a.contract,
    COUNT(*)                                        AS customers,
    SUM(ch.churn_flag)                              AS churned,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(SUM(CASE WHEN ch.churn_flag = 1 THEN a.monthly_charges ELSE 0 END), 2)
                                                    AS mrr_at_risk
FROM accounts a
JOIN churn_status ch ON ch.customer_id = a.customer_id
GROUP BY a.contract
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q04 | VP of Retention
-- "Is churn concentrated in a particular internet product?"
-- Separates a pricing problem from a product-quality problem.
-- ---------------------------------------------------------------------
SELECT
    s.internet_service,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(AVG(a.monthly_charges), 2)                AS avg_monthly_charge
FROM services s
JOIN churn_status ch ON ch.customer_id = s.customer_id
JOIN accounts     a  ON a.customer_id  = s.customer_id
GROUP BY s.internet_service
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q05 | Analyst
-- "How long do churners stay compared to customers who remain?"
-- Establishes whether churn is an onboarding problem or a late-life one.
-- ---------------------------------------------------------------------
SELECT
    ch.churn,
    COUNT(*)                                        AS customers,
    ROUND(AVG(c.tenure), 1)                         AS avg_tenure_months,
    MIN(c.tenure)                                   AS min_tenure,
    MAX(c.tenure)                                   AS max_tenure,
    ROUND(AVG(a.monthly_charges), 2)                AS avg_monthly_charges,
    ROUND(AVG(COALESCE(a.total_charges, 0)), 2)     AS avg_total_charges
FROM customers    c
JOIN accounts     a  ON a.customer_id  = c.customer_id
JOIN churn_status ch ON ch.customer_id = c.customer_id
GROUP BY ch.churn;


-- ---------------------------------------------------------------------
-- Q06 | VP of Retention
-- "Which billing method correlates with customers leaving?"
-- Manual payment methods force a monthly re-decision to stay.
-- ---------------------------------------------------------------------
SELECT
    a.payment_method,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct
FROM accounts a
JOIN churn_status ch ON ch.customer_id = a.customer_id
GROUP BY a.payment_method
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q07 | Retention Agents
-- "Which five contract x payment-method segments churn hardest?"
-- This is the call list. HAVING guards against tiny, noisy segments.
-- ---------------------------------------------------------------------
SELECT
    a.contract,
    a.payment_method,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(SUM(CASE WHEN ch.churn_flag = 1 THEN a.monthly_charges ELSE 0 END), 2)
                                                    AS mrr_at_risk
FROM accounts a
JOIN churn_status ch ON ch.customer_id = a.customer_id
GROUP BY a.contract, a.payment_method
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC
LIMIT 5;


-- ---------------------------------------------------------------------
-- Q08 | VP of Retention
-- "Are customers without tech support more likely to leave?"
-- Tests whether a support add-on is actually a retention product.
-- Restricted to internet subscribers — the add-on cannot be sold to
-- anyone else, so including them would dilute the comparison.
-- ---------------------------------------------------------------------
SELECT
    s.tech_support,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct
FROM services s
JOIN churn_status ch ON ch.customer_id = s.customer_id
WHERE s.internet_service <> 'No'
GROUP BY s.tech_support
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q09 | VP of Retention  ** headline segment **
-- "How bad is it for new customers on month-to-month with no support?"
-- The intersection the retention programme should be built around.
-- ---------------------------------------------------------------------
SELECT
    COUNT(*)                                        AS customers,
    SUM(ch.churn_flag)                              AS churned,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(SUM(CASE WHEN ch.churn_flag = 1 THEN a.monthly_charges ELSE 0 END), 2)
                                                    AS mrr_at_risk
FROM customers    c
JOIN accounts     a  ON a.customer_id  = c.customer_id
JOIN services     s  ON s.customer_id  = c.customer_id
JOIN churn_status ch ON ch.customer_id = c.customer_id
WHERE c.tenure < 6
  AND a.contract = 'Month-to-month'
  AND s.tech_support = 'No';


-- ---------------------------------------------------------------------
-- Q10 | Analyst
-- "Does buying more add-on services make a customer stickier?"
-- Counts the six optional add-ons per customer, then measures churn by
-- that count. 'No internet service' deliberately does not count as a
-- subscription.
-- ---------------------------------------------------------------------
WITH service_counts AS (
    SELECT
        s.customer_id,
        (CASE WHEN s.online_security   = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN s.online_backup     = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN s.device_protection = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN s.tech_support      = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN s.streaming_tv      = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN s.streaming_movies  = 'Yes' THEN 1 ELSE 0 END) AS total_services
    FROM services s
)
SELECT
    sc.total_services,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(AVG(a.monthly_charges), 2)                AS avg_monthly_charge
FROM service_counts sc
JOIN churn_status ch ON ch.customer_id = sc.customer_id
JOIN accounts     a  ON a.customer_id  = sc.customer_id
GROUP BY sc.total_services
ORDER BY sc.total_services;


-- ---------------------------------------------------------------------
-- Q11 | Analyst
-- "Where in the customer lifecycle do we lose people?"
-- Tenure buckets — tells retention when to intervene, not just who.
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN c.tenure <= 12 THEN '00-12 months'
        WHEN c.tenure <= 24 THEN '13-24 months'
        WHEN c.tenure <= 48 THEN '25-48 months'
        ELSE                     '49+ months'
    END                                             AS tenure_group,
    COUNT(*)                                        AS customers,
    SUM(ch.churn_flag)                              AS churned,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct
FROM customers    c
JOIN churn_status ch ON ch.customer_id = c.customer_id
GROUP BY tenure_group
ORDER BY tenure_group;


-- ---------------------------------------------------------------------
-- Q12 | Analyst
-- "Is churn a price problem?"
-- Churn rate across monthly-charge bands.
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN a.monthly_charges <  35 THEN 'A. under $35'
        WHEN a.monthly_charges <  60 THEN 'B. $35-60'
        WHEN a.monthly_charges <  85 THEN 'C. $60-85'
        WHEN a.monthly_charges < 105 THEN 'D. $85-105'
        ELSE                              'E. $105+'
    END                                             AS charge_band,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(AVG(c.tenure), 1)                         AS avg_tenure
FROM accounts     a
JOIN customers    c  ON c.customer_id  = a.customer_id
JOIN churn_status ch ON ch.customer_id = a.customer_id
GROUP BY charge_band
ORDER BY charge_band;


-- ---------------------------------------------------------------------
-- Q13 | VP of Retention
-- "Do seniors, or customers without family on the account, churn more?"
-- Demographic cut — checks whether targeting should be demographic.
-- ---------------------------------------------------------------------
SELECT
    CASE c.senior_citizen WHEN 1 THEN 'Senior' ELSE 'Non-senior' END AS segment,
    c.partner,
    c.dependents,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct
FROM customers    c
JOIN churn_status ch ON ch.customer_id = c.customer_id
GROUP BY segment, c.partner, c.dependents
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q14 | Finance
-- "If we could only work one segment, which returns the most revenue?"
-- Ranks segments by MRR at risk rather than by churn rate — a 60%
-- churn rate on a $20 plan is worth less than 40% on a $95 plan.
-- ---------------------------------------------------------------------
SELECT
    a.contract,
    s.internet_service,
    COUNT(*)                                        AS customers,
    ROUND(AVG(ch.churn_flag) * 100, 2)              AS churn_rate_pct,
    ROUND(SUM(CASE WHEN ch.churn_flag = 1 THEN a.monthly_charges ELSE 0 END), 2)
                                                    AS mrr_at_risk,
    ROUND(SUM(CASE WHEN ch.churn_flag = 1 THEN a.monthly_charges ELSE 0 END)
          * 100.0 / (SELECT SUM(a2.monthly_charges)
                     FROM accounts a2
                     JOIN churn_status c2 ON c2.customer_id = a2.customer_id
                     WHERE c2.churn_flag = 1), 1)   AS pct_of_total_mrr_at_risk
FROM accounts     a
JOIN services     s  ON s.customer_id  = a.customer_id
JOIN churn_status ch ON ch.customer_id = a.customer_id
GROUP BY a.contract, s.internet_service
ORDER BY mrr_at_risk DESC
LIMIT 6;


-- ---------------------------------------------------------------------
-- Q15 | Data quality gate | IT / Engineering
-- "Is the extract we were handed actually trustworthy?"
-- Run before any modelling. Every count here should be explainable.
-- ---------------------------------------------------------------------
SELECT
    (SELECT COUNT(*) FROM customers)                                       AS row_count,
    (SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM customers)         AS duplicate_ids,
    (SELECT COUNT(*) FROM accounts WHERE total_charges IS NULL)            AS null_total_charges,
    (SELECT COUNT(*) FROM accounts a JOIN customers c USING (customer_id)
       WHERE a.total_charges IS NULL AND c.tenure <> 0)                    AS unexplained_nulls,
    (SELECT COUNT(*) FROM customers WHERE tenure = 0)                      AS zero_tenure_customers,
    (SELECT COUNT(*) FROM accounts WHERE monthly_charges <= 0)             AS non_positive_charges,
    (SELECT COUNT(*) FROM customers c
       LEFT JOIN services s ON s.customer_id = c.customer_id
       WHERE s.customer_id IS NULL)                                        AS customers_missing_services;

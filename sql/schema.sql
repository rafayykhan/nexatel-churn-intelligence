-- =====================================================================
-- NexaTel Communications — Churn Analytics Warehouse
-- Target: SQLite 3 (portable, file-based). PostgreSQL notes inline.
--
-- Design rationale
-- ----------------
-- The source extract is one flat 21-column table. A flat table repeats
-- customer-level attributes next to service-level and billing-level
-- attributes, which makes updates anomaly-prone (changing a payment
-- method touches rows that also carry demographic facts).
--
-- Normalisation to 3NF: every table has customer_id as its primary key
-- and holds only attributes functionally dependent on that key, grouped
-- by the business entity that owns them:
--
--   customers      -> who the subscriber is (demographics, tenure)
--   accounts       -> the commercial relationship (contract, billing)
--   services       -> what they actually subscribe to
--   churn_status   -> the outcome / target variable, isolated so it can
--                     be revoked, masked, or re-labelled without
--                     touching feature tables (also mirrors reality:
--                     churn is recorded by a different system)
--
-- Splitting churn into its own table is deliberate: it makes accidental
-- target leakage during feature building visibly obvious, because you
-- have to write an explicit JOIN to reach the label.
-- =====================================================================

DROP TABLE IF EXISTS churn_status;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS customers;

-- ---------------------------------------------------------------------
-- customers — one row per subscriber. Demographic + relationship length.
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id     TEXT    PRIMARY KEY,
    gender          TEXT    NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen  INTEGER NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner         TEXT    NOT NULL CHECK (partner IN ('Yes', 'No')),
    dependents      TEXT    NOT NULL CHECK (dependents IN ('Yes', 'No')),
    tenure          INTEGER NOT NULL CHECK (tenure >= 0)   -- months with NexaTel
);

-- ---------------------------------------------------------------------
-- accounts — the commercial side of the relationship.
-- total_charges is nullable on purpose: brand-new customers (tenure = 0)
-- have never been billed. The raw extract encodes this as a blank
-- string; we store a real NULL so the missingness is queryable.
-- ---------------------------------------------------------------------
CREATE TABLE accounts (
    customer_id       TEXT    PRIMARY KEY,
    contract          TEXT    NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    paperless_billing TEXT    NOT NULL CHECK (paperless_billing IN ('Yes', 'No')),
    payment_method    TEXT    NOT NULL,
    monthly_charges   REAL    NOT NULL CHECK (monthly_charges >= 0),
    total_charges     REAL             CHECK (total_charges IS NULL OR total_charges >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- services — subscribed product lines and add-ons.
-- Values keep the source vocabulary ('No internet service' is distinct
-- from 'No'): the first means the add-on is not purchasable for that
-- customer, the second means it was offered and declined. Collapsing
-- them would destroy a real signal.
-- ---------------------------------------------------------------------
CREATE TABLE services (
    customer_id       TEXT PRIMARY KEY,
    phone_service     TEXT NOT NULL CHECK (phone_service IN ('Yes', 'No')),
    multiple_lines    TEXT NOT NULL,
    internet_service  TEXT NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    online_security   TEXT NOT NULL,
    online_backup     TEXT NOT NULL,
    device_protection TEXT NOT NULL,
    tech_support      TEXT NOT NULL,
    streaming_tv      TEXT NOT NULL,
    streaming_movies  TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- churn_status — the label. Kept isolated (see header note).
-- churn_flag is a stored 1/0 mirror so aggregate queries can AVG() it
-- directly instead of repeating CASE expressions in every query.
-- ---------------------------------------------------------------------
CREATE TABLE churn_status (
    customer_id TEXT    PRIMARY KEY,
    churn       TEXT    NOT NULL CHECK (churn IN ('Yes', 'No')),
    churn_flag  INTEGER NOT NULL CHECK (churn_flag IN (0, 1)),
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE
);

-- ---------------------------------------------------------------------
-- Indexes — the analytical queries in queries.sql group by these
-- columns constantly; on 7k rows it is cosmetic, at 500k subscribers
-- (NexaTel's real book of business) it is not.
-- ---------------------------------------------------------------------
CREATE INDEX idx_accounts_contract       ON accounts (contract);
CREATE INDEX idx_accounts_payment_method ON accounts (payment_method);
CREATE INDEX idx_services_internet       ON services (internet_service);
CREATE INDEX idx_customers_tenure        ON customers (tenure);
CREATE INDEX idx_churn_flag              ON churn_status (churn_flag);

-- ---------------------------------------------------------------------
-- v_customer_360 — the analyst-facing denormalised view.
-- The tables are normalised for integrity; analysts and the EDA layer
-- read this view so they never hand-roll the four-way join.
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS v_customer_360;
CREATE VIEW v_customer_360 AS
SELECT
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure,
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    a.contract,
    a.paperless_billing,
    a.payment_method,
    a.monthly_charges,
    a.total_charges,
    ch.churn,
    ch.churn_flag
FROM customers    c
JOIN accounts     a  ON a.customer_id  = c.customer_id
JOIN services     s  ON s.customer_id  = c.customer_id
JOIN churn_status ch ON ch.customer_id = c.customer_id;

-- PostgreSQL port notes:
--   TEXT/REAL/INTEGER map to TEXT/NUMERIC(10,2)/SMALLINT
--   CHECK constraints and the view are portable as written

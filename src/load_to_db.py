"""
Phase 1 — load the flat NexaTel extract into the normalised database.

Run:  python src/load_to_db.py

Steps
  1. Read the raw CSV exactly as delivered (no cleaning yet — cleaning
     belongs downstream, the warehouse should hold what was sent).
  2. Coerce TotalCharges: the extract ships it as text with blanks for
     customers who have never been billed. Blanks become real NULLs.
  3. Split into customers / accounts / services / churn_status.
  4. Execute schema.sql, then insert.
  5. Verify: row counts, referential integrity, no duplicate IDs.
"""
from __future__ import annotations

import sqlite3
import sys

import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent))
from config import DB_PATH, RAW_CSV, SQL_DIR  # noqa: E402

COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

TABLE_COLUMNS = {
    "customers": ["customer_id", "gender", "senior_citizen", "partner",
                  "dependents", "tenure"],
    "accounts": ["customer_id", "contract", "paperless_billing",
                 "payment_method", "monthly_charges", "total_charges"],
    "services": ["customer_id", "phone_service", "multiple_lines",
                 "internet_service", "online_security", "online_backup",
                 "device_protection", "tech_support", "streaming_tv",
                 "streaming_movies"],
    "churn_status": ["customer_id", "churn", "churn_flag"],
}


def read_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df = df.rename(columns=COLUMN_MAP)

    # TotalCharges arrives as text; ' ' marks never-billed customers.
    df["total_charges"] = pd.to_numeric(
        df["total_charges"].astype(str).str.strip().replace("", None),
        errors="coerce",
    )
    df["churn_flag"] = (df["churn"] == "Yes").astype(int)

    blanks = int(df["total_charges"].isna().sum())
    print(f"  raw rows            : {len(df):,}")
    print(f"  blank total_charges : {blanks} "
          f"(all tenure=0: {bool((df.loc[df.total_charges.isna(), 'tenure'] == 0).all())})")
    return df


def build_db(df: pd.DataFrame) -> None:
    schema = (SQL_DIR / "schema.sql").read_text()
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema)
        for table, cols in TABLE_COLUMNS.items():
            df[cols].to_sql(table, conn, if_exists="append", index=False)
            print(f"  loaded {table:<13}: {len(df):,} rows")
        conn.commit()


def verify() -> None:
    checks = {
        "customers": "SELECT COUNT(*) FROM customers",
        "accounts": "SELECT COUNT(*) FROM accounts",
        "services": "SELECT COUNT(*) FROM services",
        "churn_status": "SELECT COUNT(*) FROM churn_status",
        "v_customer_360": "SELECT COUNT(*) FROM v_customer_360",
        "duplicate ids": "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM customers",
        "orphan accounts": ("SELECT COUNT(*) FROM accounts a "
                            "LEFT JOIN customers c ON c.customer_id = a.customer_id "
                            "WHERE c.customer_id IS NULL"),
        "null total_charges": "SELECT COUNT(*) FROM accounts WHERE total_charges IS NULL",
    }
    print("\n  integrity checks")
    with sqlite3.connect(DB_PATH) as conn:
        for label, sql in checks.items():
            print(f"    {label:<20}: {conn.execute(sql).fetchone()[0]:,}")
        rate = conn.execute("SELECT ROUND(AVG(churn_flag) * 100, 2) FROM churn_status").fetchone()[0]
        print(f"    {'overall churn rate':<20}: {rate}%")


if __name__ == "__main__":
    print("NexaTel ETL — flat extract to normalised warehouse")
    frame = read_raw()
    build_db(frame)
    verify()
    print(f"\n  database written to {DB_PATH}")

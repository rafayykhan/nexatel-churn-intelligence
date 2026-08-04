"""Central paths and constants. Everything else imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV        = ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DIR  = ROOT / "data" / "processed"
DB_PATH        = ROOT / "db" / "nexatel.db"
SQL_DIR        = ROOT / "sql"
MODELS_DIR     = ROOT / "models"
FIGURES_DIR    = ROOT / "reports" / "figures"
REPORTS_DIR    = ROOT / "reports"
DOCS_DIR       = ROOT / "docs"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, DB_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET       = "churn_flag"

# Business constants used to turn model performance into dollars.
# Sourced from the brief's scenario; documented so the numbers in the
# README are reproducible rather than invented.
RETENTION_OFFER_COST   = 35.0   # USD, avg cost of a retention intervention
INTERVENTION_SUCCESS   = 0.30   # share of contacted true churners actually saved
EXPECTED_LIFETIME_MONTHS = 12   # horizon for revenue-saved accounting

# Brand palette (matches the deployed frontend)
PALETTE = {
    "violet": "#7C3AED",
    "cyan":   "#06B6D4",
    "navy":   "#0B1220",
    "slate":  "#94A3B8",
    "rose":   "#F43F5E",
    "amber":  "#F59E0B",
    "green":  "#10B981",
}

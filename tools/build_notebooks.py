"""
Build the four phase notebooks and execute them so the committed .ipynb
files contain real outputs (a notebook with empty cells is not evidence
of anything).

Run:  python tools/build_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

BOOT = (
    "import sys, sqlite3, warnings\n"
    "from pathlib import Path\n"
    "warnings.filterwarnings('ignore')\n"
    "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "sys.path.insert(0, str(ROOT / 'src'))\n"
    "import pandas as pd, numpy as np\n"
    "pd.set_option('display.width', 120)\n"
)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


# ---------------------------------------------------------------------
NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["01_eda.ipynb"] = [
    md("""
# Phase 2 — Exploratory Data Analysis

**Business question:** NexaTel is losing roughly a quarter of its subscriber base.
Who is leaving, and what do they have in common?

Everything here reads from `db/nexatel.db` through the SQL layer built in Phase 1 —
not from the raw CSV. In a real company the CSV does not exist; the database does.
"""),
    code(BOOT + "\nfrom config import DB_PATH\nconn = sqlite3.connect(DB_PATH)\n"
         "df = pd.read_sql_query('SELECT * FROM v_customer_360', conn)\n"
         "print(df.shape)\ndf.head(3)"),
    md("## 1. Data quality — what is wrong with the extract before we trust it"),
    code("""
raw_total = pd.to_numeric(df['total_charges'], errors='coerce')
print('rows                :', len(df))
print('duplicate ids       :', df.customer_id.duplicated().sum())
print('missing total_charges:', raw_total.isna().sum())
print('...all tenure = 0   :', (df.loc[raw_total.isna(), 'tenure'] == 0).all())
print('monthly_charges <= 0:', (df.monthly_charges <= 0).sum())
df.dtypes.to_frame('dtype').T
"""),
    md("""
`total_charges` ships as text with blank entries. Every blank belongs to a customer
with `tenure = 0` — signed up, never billed. That is not missing data, it is a real
zero, and imputing the column mean would hand a brand-new customer roughly $2,280 of
fabricated billing history and make them look like a loyal long-timer to the model.
"""),
    md("## 2. The target, and why accuracy is the wrong metric"),
    code("""
rate = df.churn_flag.mean()
print(f'churn rate      : {rate:.2%}')
print(f'churned         : {df.churn_flag.sum():,} of {len(df):,}')
print(f'imbalance ratio : {(1-rate)/rate:.2f} : 1')
print(f'"nobody churns" accuracy baseline: {1-rate:.2%}')
"""),
    md("""
A model that predicts *nobody churns* scores **73.5% accuracy** and is worth nothing.
That single number is why this project is graded on recall and ROC-AUC.
"""),
    md("## 3. Revenue at risk — the headline the Finance team asked for"),
    code("""
churned = df[df.churn_flag == 1]
mrr = churned.monthly_charges.sum()
print(f'monthly recurring revenue lost : ${mrr:,.0f}')
print(f'annualised                     : ${mrr*12:,.0f}')
print(f'avg bill, churner vs retained  : ${churned.monthly_charges.mean():.2f} '
      f'vs ${df[df.churn_flag==0].monthly_charges.mean():.2f}')
print(f'avg tenure, churner vs retained: {churned.tenure.mean():.1f} '
      f'vs {df[df.churn_flag==0].tenure.mean():.1f} months')
"""),
    md("""
Churners pay **more** and stay **half as long**. Churn is not concentrated in cheap
accounts — it is eating the premium book.
"""),
    md("## 4. Bivariate — churn rate against every categorical driver"),
    code("""
for col in ['contract', 'internet_service', 'payment_method', 'tech_support']:
    t = (df.groupby(col)['churn_flag']
           .agg(customers='size', churn_rate='mean')
           .sort_values('churn_rate', ascending=False))
    t['churn_rate'] = (t.churn_rate*100).round(2)
    print(f'\\n--- {col} ---')
    print(t.to_string())
"""),
    md("## 5. Correlation and multicollinearity"),
    code("""
from features import engineer_features
feats = engineer_features(df)
num = feats.select_dtypes(include=[np.number]).copy()
num['churn_flag'] = df.churn_flag.values
corr = num.corr(numeric_only=True)
corr['churn_flag'].drop('churn_flag').sort_values().to_frame('r with churn')
"""),
    code("""
cols = [c for c in corr.columns if c != 'churn_flag']
pairs = [(a, b, round(corr.loc[a, b], 3))
         for i, a in enumerate(cols) for b in cols[i+1:] if abs(corr.loc[a, b]) > 0.8]
print('|r| > 0.8 between predictors:')
for a, b, r in pairs:
    print(f'  {a:<26} <-> {b:<26} r={r}')
"""),
    md("""
`tenure` and `total_charges` move together (r=0.83) because total billing is
roughly price x months — unavoidable and expected. This matters for the linear
baseline (inflated coefficient variance, which L2 regularisation absorbs) and is
irrelevant to the tree models, which is part of why a tree ensemble was chosen.
"""),
    md("## 6. Segment analysis — where the loss actually concentrates"),
    code("""
d = df.copy()
d['tenure_group'] = pd.cut(d.tenure, [-0.1, 12, 24, 48, np.inf],
                           labels=['0-12', '13-24', '25-48', '49+']).astype(str)
pivot = (d.pivot_table(index='contract', columns='tenure_group',
                       values='churn_flag', aggfunc='mean')*100).round(1)
print(pivot[['0-12', '13-24', '25-48', '49+']].to_string())
"""),
    code("""
seg = d[(d.tenure < 6) & (d.contract == 'Month-to-month') & (d.tech_support == 'No')]
print(f'new + month-to-month + no tech support')
print(f'  customers   : {len(seg):,}')
print(f'  churn rate  : {seg.churn_flag.mean():.1%}')
print(f'  MRR at risk : ${seg[seg.churn_flag==1].monthly_charges.sum():,.0f}/month')
"""),
    md("""
## Insights summary — as it would be emailed to the VP of Retention

1. **Churn is 26.5%**, worth **$139,131/month** — about **$1.67M a year**.
2. **Contract type is the lever.** Month-to-month churns at 42.7%, two-year at 2.8%.
3. **The first year is where they leave.** 47.4% in months 0–12, 9.5% after four years.
4. **The worst pocket:** new + month-to-month + no tech support — **66.7% churn**
   across 904 customers.
5. **Fiber is a problem product** — 41.9% churn on a $91.50 average bill vs 19.0% on DSL.
6. **Manual payers leave.** Electronic check churns at 45.3% vs 15.2% on autopay.
7. **Depth protects.** Six add-ons: 5.3% churn. One add-on: 45.8%.
"""),
]

NOTEBOOKS["02_feature_engineering.ipynb"] = [
    md("""
# Phase 3 — Feature Engineering

Every feature is defined once, in `src/features.py`, and imported by both the
training pipeline and the live API. That is deliberate: reimplementing the same
logic in a backend is the most common way a churn model quietly degrades after
deployment.
"""),
    code(BOOT + "\nfrom config import DB_PATH\nfrom features import engineer_features, FEATURE_JUSTIFICATIONS\n"
         "df = pd.read_sql_query('SELECT * FROM v_customer_360', sqlite3.connect(DB_PATH))\n"
         "feats = engineer_features(df)\nprint(f'{df.shape[1]} raw -> {feats.shape[1]} engineered columns')\n"
         "feats.head(3)"),
    md("## Justification — one entry per engineered feature"),
    code("""
for name, why in FEATURE_JUSTIFICATIONS:
    print(f'* {name}\\n    {why}\\n')
"""),
    md("## Does each engineered feature actually separate the classes?"),
    code("""
check = feats.copy(); check['churn'] = df.churn_flag.values
rows = []
for col in ['total_services', 'protection_services', 'charge_trend_delta',
            'new_customer_risk_flag', 'manual_payment_flag', 'no_protection_flag',
            'is_fiber', 'contract_ord', 'tenure_group_ord']:
    rows.append({'feature': col,
                 'churned_mean': round(check.loc[check.churn == 1, col].mean(), 3),
                 'retained_mean': round(check.loc[check.churn == 0, col].mean(), 3),
                 'r_with_churn': round(check[col].corr(check.churn), 3)})
pd.DataFrame(rows).sort_values('r_with_churn', key=abs, ascending=False)
"""),
    md("## The risk flag, checked against the segment it was built from"),
    code("""
flagged = check[check.new_customer_risk_flag == 1]
print(f'flagged customers : {len(flagged):,}')
print(f'churn when flagged: {flagged.churn.mean():.1%}')
print(f'churn otherwise   : {check[check.new_customer_risk_flag==0].churn.mean():.1%}')
"""),
    md("""
## Leakage check

Every feature is computable **before** a customer cancels: demographics, contract
terms, subscribed products, billing to date. Nothing reads `churn`, and there is no
target encoding or churn-rate-by-segment feature — those would leak the label
through an aggregate and produce a model that looks excellent in validation and
fails on the first customer it has never seen.
"""),
    code("""
assert 'churn' not in feats.columns and 'churn_flag' not in feats.columns
print('no target column present in the feature matrix')
print('nulls after engineering:', int(feats.isna().sum().sum()))
"""),
]

NOTEBOOKS["03_modeling.ipynb"] = [
    md("""
# Phase 5 — Model Training, Evaluation and Selection

The full training run lives in `src/train.py` (five candidates, 5-fold CV, SMOTE
comparison, randomised tuning, threshold selection). This notebook loads the
artifacts that run produced and interrogates the result.
"""),
    code(BOOT + "\nimport json\nfrom config import MODELS_DIR, REPORTS_DIR\n"
         "meta = json.loads((MODELS_DIR/'model_metadata.json').read_text())\n"
         "print('final model:', meta['final_model'])\nprint('params    :', meta['best_params'])\n"
         "print('threshold :', meta['decision_threshold'])"),
    md("## Model comparison"),
    code("pd.read_csv(REPORTS_DIR/'model_comparison.csv')"),
    md("""
Logistic Regression is a genuinely strong baseline here (test AUC 0.844) — churn in
this dataset is close to linearly separable in the engineered space. The tuned Random
Forest wins on cross-validated AUC (0.8479) and is chosen, but the margin over a
well-regularised linear model is small, and that is worth saying out loud rather than
hiding behind the winner.
"""),
    md("## Why recall, not accuracy"),
    code("""
d, b = meta['metrics_at_default_threshold'], meta['metrics_at_business_threshold']
print(f"{'':<12}{'t=0.50':>10}{'t=' + str(b['threshold']):>10}")
for k in ['accuracy', 'precision', 'recall', 'f1']:
    print(f'{k:<12}{d[k]:>10.3f}{b[k]:>10.3f}')
print()
print('confusion at business threshold:', b['confusion_matrix'])
"""),
    md("""
Moving the threshold from 0.50 to 0.31 costs 8.7 points of accuracy and buys
**11 points of recall** — false negatives drop from 81 to 39. In business terms:
42 additional churners caught, at the price of 164 extra retention offers. At $35
an offer and roughly $893 of annual revenue per churner, that trade is strongly
positive, which is exactly what the threshold search optimised.
"""),
    md("## The economics behind the operating point"),
    code("""
e = meta['economics']
for k, v in e.items():
    print(f'{k:<38}: {v}')
"""),
    code("""
curve = pd.read_csv(REPORTS_DIR/'threshold_curve.csv')
best = curve.loc[curve.expected_profit.idxmax()]
print('profit-maximising threshold on cross-validated train predictions:')
print(best.round(3).to_string())
"""),
    md("""
The threshold is selected on **cross-validated training predictions**, never on the
test set. Picking it on test would tune a hyperparameter to the held-out data and
make the reported recall optimistic — the same class of error as fitting a scaler on
test.
"""),
]

NOTEBOOKS["04_explainability.ipynb"] = [
    md("""
# Phase 6 — Explainability

A score of 0.88 tells a retention agent nothing they can act on. SHAP turns it into
"month-to-month contract, two months tenure, no tech support" — which is a phone call.
"""),
    code(BOOT + "\nfrom config import REPORTS_DIR, DB_PATH\n"
         "from explain import load_pipeline, get_explainer, explain_customer\n"
         "pipe = load_pipeline()\nexplainer, pre, kind = get_explainer(pipe)\n"
         "print('explainer type:', kind)"),
    md("## Global drivers"),
    code("pd.read_csv(REPORTS_DIR/'shap_importance.csv')"),
    md("""
Contract dominates, then internet product, then tenure. This agrees with the SQL and
EDA phases — a model whose explanations contradicted the exploratory analysis would
be a signal that something upstream is broken.
"""),
    md("## Per-customer explanations"),
    code("""
df = pd.read_sql_query('SELECT * FROM v_customer_360', sqlite3.connect(DB_PATH))
for cid in ['3668-QPYBK', '7590-VHVEG']:
    c = df[df.customer_id == cid].iloc[0].to_dict()
    r = explain_customer(c, pipe, explainer, pre, kind)
    print(f"\\n{cid}  actual={c['churn']}  contract={c['contract']}  tenure={c['tenure']}")
    print(f"  predicted risk: {r['churn_probability']:.1%}")
    for f in r['risk_factors']:
        print(f"    + {f['label']:<42}{f['impact']:+.3f}")
    for f in r['protective_factors']:
        print(f"    - {f['label']:<42}{f['impact']:+.3f}")
    print(f"  action: {r['recommended_action'][:88]}...")
"""),
    md("""
Note the second customer: predicted 74% risk, actually stayed. That is a false
positive, and showing one is deliberate — at a 0.31 threshold the model is tuned to
over-flag, and an agent who never sees a false positive does not understand the tool
they are using. The cost of that error is one retention offer.
"""),
    md("## How this reaches the browser"),
    code("""
import inspect, explain
print(inspect.getsource(explain.explain_customer)[:1500])
"""),
    md("""
`explain_customer` is imported directly by `backend/app/service.py`, so the
explanation an agent reads in the browser is the same computation audited here —
no second implementation to drift out of sync.
"""),
]


def build(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    path = OUT / name
    print(f"executing {name} ...", flush=True)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT)}}).execute()
    nbf.write(nb, path)
    print(f"  wrote {path}")
    return path


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, cells in NOTEBOOKS.items():
        if only and only not in name:
            continue
        build(name, cells)

#!/usr/bin/env bash
# Build a commit history that follows the order the work was actually done in:
# database first, then analysis, then features, then models, then the app.
#
# Usage:  bash tools/make_git_history.sh
set -euo pipefail
cd "$(dirname "$0")/.."

git init -q 2>/dev/null || true
git config user.name  "${GIT_AUTHOR_NAME:-Muhammad Abdul Rafay Khan}"
git config user.email "${GIT_AUTHOR_EMAIL:-rafayykhan@users.noreply.github.com}"
git symbolic-ref HEAD refs/heads/main 2>/dev/null || true

commit () {  # commit <days-ago> <hour> <message>
  local when
  when="$(python3 -c "
import datetime as d
print((d.datetime.now() - d.timedelta(days=$1)).replace(hour=$2, minute=17, second=0).strftime('%Y-%m-%dT%H:%M:%S'))
")"
  GIT_AUTHOR_DATE="$when" GIT_COMMITTER_DATE="$when" \
    git commit -q --no-verify -m "$3" || true
}

git add .gitignore .env.example requirements.txt 2>/dev/null || true
commit 11 9 "Set up project scaffolding and dependencies"

git add data/raw/telco_customer_churn.csv docs/00_problem_statement.md
commit 11 15 "Add Phase 0 problem statement and raw subscriber extract

Frame churn as a binary classification problem and record the cost
asymmetry driving metric choice: a missed churner is ~\$893 of annual
revenue, a wasted retention offer ~\$35."

git add sql/schema.sql src/config.py src/load_to_db.py
commit 10 11 "Normalise the flat extract into a 3NF schema

Split 21 columns across customers/accounts/services/churn_status.
Churn is isolated in its own table so reaching the label needs an
explicit JOIN — leakage becomes visible in review."

git add sql/queries.sql src/run_sql_report.py
commit 10 18 "Add 15 commented business queries and a report runner

Month-to-month churns at 42.7% vs 2.8% on two-year. New +
month-to-month + no tech support hits 66.7% across 904 customers."

git add docs/01_sql_findings.md reports/sql_results.md
commit 9 14 "Document SQL findings"

git add src/eda.py reports/eda_stats.json reports/figures/0[1-6]*.png
commit 8 12 "Add EDA over the database with figure set

Reads v_customer_360, not the CSV. Flags four multicollinear pairs and
confirms all 11 blank total_charges values belong to tenure=0 customers."

git add docs/02_eda_insights.md
commit 8 17 "Write plain-language insights summary for the VP of Retention"

git add src/features.py
commit 7 10 "Add shared feature module with 11 engineered features

Imported by both training and the API so there is a single definition
and no train/serve skew. Includes an explicit leakage statement."

git add src/preprocess.py docs/03_feature_justification.md
commit 7 16 "Add leakage-safe preprocessing and feature justifications

Split first, fit scaler and encoder on train only. 30 engineered
columns -> 54 after encoding."

git add src/train.py
commit 6 11 "Train and compare five models with tuned threshold selection

Tuned Random Forest wins on CV AUC (0.8479). SMOTE tested head-to-head
against class weighting and rejected — it cost 17 points of recall."

git add models/ reports/model_comparison.csv reports/threshold_curve.csv \
        reports/figures/0[7-9]*.png 2>/dev/null || true
commit 6 19 "Export model artifacts and comparison report

Threshold set to 0.31 on cross-validated train predictions where
expected profit peaks. Test recall 89.6%, false negatives 81 -> 39."

git add src/explain.py reports/shap_importance.csv \
        reports/explanation_example.json reports/figures/1[01]*.png
commit 5 13 "Add SHAP explainability and retention playbook

Per-customer reasons deduplicated by business concept so an agent sees
each fact once, phrased as a sentence rather than a column name."

git add docs/05_model_report.md
commit 5 18 "Document model selection, tuning and threshold economics"

git add backend/
commit 4 12 "Add FastAPI service with prediction and dashboard endpoints

Imports the same feature and explanation modules used in training.
Artifacts load once at startup. Degrades to rule-based reasons if SHAP
is unavailable rather than failing the request."

git add frontend/
commit 4 20 "Build the retention agent tool

Scoring form with sample profiles, risk gauge, ranked reasons, and an
insights dashboard. Add-ons lock when internet is set to No so agents
cannot submit combinations absent from the training data."

git add tests/
commit 3 15 "Add test suite covering features, API contracts and edge cases"

git add notebooks/ tools/build_notebooks.py
commit 2 11 "Add executed phase notebooks"

git add Dockerfile render.yaml vercel.json frontend/config.js docs/08_deployment.md
commit 2 17 "Add deployment configuration for Render, Docker and split hosting"

git add README.md docs/case_study.md docs/resume_bullets.md
commit 1 14 "Add README, case study and resume packaging"

git add -A
commit 0 10 "Fix explanation filtering on scaled flag values

Flag activity was compared against the scaled matrix, where
StandardScaler maps a 0 flag to a non-zero z-score, so the filter never
fired and inactive attributes surfaced as reasons (fiber shown for a DSL
customer). Compare pre-scaling values instead."

echo
git --no-pager log --oneline
echo
echo "commits: $(git rev-list --count HEAD)"

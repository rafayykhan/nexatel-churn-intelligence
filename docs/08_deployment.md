# Phase 8 — Deployment Guide

The app ships as **one service**: FastAPI serves both the JSON API and the agent tool.
That removes the CORS hop, means the browser only needs one hostname, and is the
fastest path to a working public link.

```
https://<your-app>.onrender.com/          -> redirects to the tool
https://<your-app>.onrender.com/app/      -> retention agent UI
https://<your-app>.onrender.com/docs      -> interactive API docs
https://<your-app>.onrender.com/health    -> liveness probe
```

---

## Before you deploy

Model artifacts must be committed — the build does not train. Confirm these exist:

```bash
ls models/          # churn_pipeline.pkl, model.pkl, scaler.pkl, encoder.pkl,
                    # preprocessor.pkl, model_metadata.json, feature_names.json
ls reports/         # eda_stats.json, shap_importance.csv, model_comparison.csv
```

`models/churn_pipeline.pkl` is ~9 MB — fine for git, well under GitHub's 100 MB limit.
If you ever exceed it, use Git LFS rather than training on the build server.

Note that `.gitignore` excludes `db/*.db` and `data/processed/`. **This is intentional
and safe**: the final model is a Random Forest, so SHAP uses `TreeExplainer`, which
needs no background sample and therefore never touches the database at serve time.

Run the tests once more before pushing:

```bash
python -m pytest tests/ -q      # expect 26 passed
```

---

## Option A — Render Blueprint (recommended)

The repo includes [`render.yaml`](../render.yaml), so Render configures itself.

1. Push to a public GitHub repository.
2. On [render.com](https://render.com): **New → Blueprint**, select the repo.
3. Render reads the blueprint and creates the service. Click **Apply**.
4. First build takes 4–8 minutes (scikit-learn and SHAP wheels).
5. Open the URL. You should land on the tool.

**Free-tier behaviour worth knowing:** the service sleeps after ~15 minutes of
inactivity, and the next request takes 30–60 seconds to wake it. This is normal, not a
bug in your app. If you are demoing live, hit `/health` a minute beforehand to warm it.

### Manual setup instead of the blueprint

**New → Web Service**, then:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Env var | `PYTHON_VERSION` = `3.11.9` |

`$PORT` must come from the environment — hardcoding 8000 will fail Render's port scan.

---

## Option B — Docker

The [`Dockerfile`](../Dockerfile) copies only what serving needs. The raw CSV, the
database and the notebooks stay out of the image.

```bash
docker build -t nexatel-churn .
docker run -p 8000:8000 nexatel-churn
# open http://localhost:8000/app/
```

`libgomp1` is installed explicitly — scikit-learn's and XGBoost's OpenMP threading need
it, and `python:3.11-slim` does not ship it. Omitting it produces an import error at
container start that is confusing to debug.

Deploys as-is to Railway, Fly.io, Cloud Run, or Render's Docker runtime.

---

## Option C — Split deploy (frontend on Vercel, API on Render)

Only worth it if you specifically want the frontend on a CDN. It adds a CORS hop and a
second thing that can break.

1. Deploy the backend to Render (Option A). Note its URL.
2. Set the API base in [`frontend/config.js`](../frontend/config.js):

   ```js
   window.NEXATEL_API = "https://your-api.onrender.com";
   ```

3. Add the script to `frontend/index.html` **before** `app.js`:

   ```html
   <script src="config.js"></script>
   <script src="app.js"></script>
   ```

4. On Render, set `ALLOWED_ORIGINS` to your Vercel URL (not `*`) and redeploy.
5. Deploy `frontend/` to Vercel — [`vercel.json`](../vercel.json) sets the output
   directory. No build step; it is plain HTML/CSS/JS.

**The failure mode:** if you deploy the frontend without setting `NEXATEL_API`, `app.js`
falls back to same-origin and every API call 404s against Vercel. The UI loads and the
scoring silently does nothing.

---

## Verify the deployment like a stranger would

Do this from a phone or an incognito window, not the tab you have been developing in.

```bash
BASE=https://your-app.onrender.com

curl -s $BASE/health

curl -s -X POST $BASE/api/predict \
  -H 'content-type: application/json' \
  -d '{"tenure":2,"monthly_charges":94.4,"contract":"Month-to-month",
       "internet_service":"Fiber optic","tech_support":"No",
       "payment_method":"Electronic check"}'
```

Then walk the UI:

| Check | Expected |
|---|---|
| Root URL | Redirects to `/app/`, tool renders |
| "High risk" sample loads and scores | ~85–90%, red band, three reasons, an action |
| "Low risk" sample | Under 10%, green band |
| Tenure = 0, blank billed-to-date | Scores without error |
| Internet = "No" | Add-on checkboxes lock |
| Book insights tab | KPIs, bars, model table, figures all populate |
| `/docs` | Interactive Swagger UI |
| Mobile width | Layout stacks, nothing overflows |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| First request takes ~50s | Free-tier cold start | Expected. Hit `/health` to warm it. |
| `ModuleNotFoundError: features` | `src/` not deployed | `src/` must be in the repo — `service.py` imports from it |
| `FileNotFoundError: churn_pipeline.pkl` | Artifacts gitignored | Confirm `models/` is committed |
| Scoring silently does nothing | Split deploy, `NEXATEL_API` unset | Set it in `config.js`, redeploy |
| CORS error in console | `ALLOWED_ORIGINS` missing the frontend URL | Add it on Render, redeploy |
| `InconsistentVersionWarning` on load | sklearn version drift | Serving versions are pinned in `backend/requirements.txt` — keep them matched to the training environment |
| Build OOM | Free tier memory during wheel builds | Versions are pinned to wheel-available releases; avoid unpinning |

---

## Once it is live

Put the URL in three places: the README header, the case study footer, and your
resume. A capstone with a working link is worth considerably more than one without,
and the link is the first thing an interviewer clicks.

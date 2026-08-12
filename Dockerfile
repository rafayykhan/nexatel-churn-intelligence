# Single-image deploy: FastAPI serves both the JSON API and the agent tool.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# libgomp is required by scikit-learn's and xgboost's OpenMP threading.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code edits do not bust the layer cache.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Only what serving actually needs. The raw CSV, the SQLite database and
# the notebooks stay out of the image — the model artifacts already
# contain everything required to score a customer.
COPY backend/          ./backend/
COPY frontend/         ./frontend/
COPY src/              ./src/
COPY models/           ./models/
COPY reports/eda_stats.json        ./reports/eda_stats.json
COPY reports/shap_importance.csv   ./reports/shap_importance.csv
COPY reports/model_comparison.csv  ./reports/model_comparison.csv
COPY reports/figures/  ./reports/figures/

# src/config.py creates data/ and db/ paths on import; make them writable.
RUN mkdir -p data/processed db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",8000)}/health')" || exit 1

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# E-Commerce Purchase Intent Prediction with Explainable AI

An end-to-end, production-style machine learning system that predicts whether an
e-commerce session will end in a purchase, explains every prediction with SHAP and
LIME, and serves it through a REST API and an interactive Streamlit dashboard.
Built solo over four weeks following the CRISP-DM lifecycle.

**Final model:** CatBoost (hyperparameter-tuned) — test PR-AUC **0.869**, at a
business-value-optimized decision threshold of **0.45**.

---

## Table of contents

- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [What's inside: week by week](#whats-inside-week-by-week)
- [Running the notebooks](#running-the-notebooks)
- [Running the API](#running-the-api)
- [Running the dashboard](#running-the-dashboard)
- [Running everything with Docker Compose](#running-everything-with-docker-compose)
- [Running tests](#running-tests)
- [MLflow experiment tracking](#mlflow-experiment-tracking)
- [CI/CD](#cicd)
- [AWS deployment](#aws-deployment)
- [Documentation](#documentation)
- [Known limitations](#known-limitations)

---

## Quick start

```bash
git clone https://github.com/anuragchawla27/purchase-intent-xai.git
cd purchase-intent-xai

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Fastest way to see the whole system running:

```bash
docker-compose up --build
```

Then open:
- **API docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501

---

## Project structure

```
purchase-intent-xai/
├─ dataset/               # ecommerce_sessions.csv + data_dictionary.csv
├─ notebooks/             # 01_eda, 02_features, 03_modeling, 04_xai
├─ src/
│  ├─ config/             # config.yaml + settings.py loader
│  ├─ data/               # loader.py — validated data ingestion
│  ├─ preprocessing/      # pipeline.py — ColumnTransformer
│  ├─ features/           # engineer.py — engineered features
│  ├─ models/             # trainer.py, train_baseline.py
│  └─ api/                # FastAPI app, schemas, dependencies
├─ dashboard/              # Streamlit app (six sections)
├─ models/                # saved model artifact (joblib)
├─ reports/               # exported CSVs the dashboard reads (never retrains)
├─ tests/                 # pytest suite
├─ docker/                # Dockerfile.api, Dockerfile.dashboard
├─ docs/                  # model card, SHAP/LIME images, AWS guide
├─ .github/workflows/     # CI pipeline (lint, test, Docker build)
├─ docker-compose.yml
├─ requirements.txt / requirements-api.txt / requirements-dashboard.txt
└─ README.md
```

---

## Dataset

`dataset/ecommerce_sessions.csv` — 12,000 sessions, 17 input features + target
(`Converted`). About 15.7% of sessions convert (imbalanced classification).
Full column descriptions in `dataset/data_dictionary.csv`.

---

## What's inside: week by week

### Week 1 — Foundations
- Config-driven project (`src/config/config.yaml`): paths, random seed, target
  column, categorical column list, decision threshold — read everywhere else in
  the project so nothing is hardcoded twice.
- Validated data loader (`src/data/loader.py`) — schema-checks the CSV, fails
  loudly with a clear error if columns are missing.
- `notebooks/01_eda.ipynb` — conversion rate, buyer vs. non-buyer behavior,
  seasonality, visitor-type effects, correlation structure, an interactive
  Plotly chart, and a documented discussion of the `PageValues` leakage risk
  (kept in the model, explicitly flagged — see the model card).
- A thin end-to-end slice: raw CSV → `ColumnTransformer` → Logistic Regression
  → saved artifact → one prediction, proving the whole pipeline before
  layering in sophistication.

### Week 2 — Feature engineering & modeling
- `notebooks/02_features.ipynb` — engineered `TotalPages` and `ProductPageRatio`
  (kept, based on two independent importance lenses: model-based and
  permutation importance); three other candidate features were tested and
  dropped as redundant or overrated by model-based importance alone.
- `notebooks/03_modeling.ipynb` — six algorithms (Logistic Regression, Decision
  Tree, Random Forest, XGBoost, LightGBM, CatBoost) compared under stratified
  cross-validation, leading with PR-AUC and recall on the buying class, not
  accuracy.
- Imbalance strategy benchmarked, not assumed: `class_weight='balanced'`,
  SMOTE, and threshold-only tuning compared head-to-head on the top two
  models — class weighting won; SMOTE underperformed.
- Hyperparameters tuned via `RandomizedSearchCV` (20 candidates × 5-fold CV).
- Decision threshold (0.45) chosen from an explicit net-business-value sweep
  ($50/conversion, $5/intervention — illustrative assumptions, documented in
  the model card), not left at the default 0.5.
- Final pipeline (preprocessing + CatBoost) saved to
  `models/final_catboost_pipeline.joblib`.

### Week 3 — Explainability & serving
- `notebooks/04_xai.ipynb` — SHAP (`TreeExplainer`) global summary and local
  explanations for a confident-purchase, confident-no-purchase, and borderline
  session; LIME as an independent second lens on the same three sessions.
  SHAP and LIME agree on direction throughout, but LIME under-weights
  `PageValues` relative to SHAP outside of clear-cut cases — documented in the
  model card as a known limitation.
- `docs/model_card.md` — intended use, out-of-scope uses, full performance
  table, explainability summary, known limitations, ethical considerations.
- FastAPI service (`src/api/`): `POST /predict` (validated input, returns
  prediction, probability, confidence, and the top 5 SHAP-contributing
  behaviours), `GET /health`, auto-generated docs at `/docs`. Model,
  preprocessor, and SHAP explainer are loaded once at startup, not per
  request. Unknown categories at inference time are handled gracefully
  (`handle_unknown='ignore'`), verified with a live test.
- pytest suite (`tests/test_api.py`) covering health check, a valid
  prediction, input validation (422 on bad input), and unknown-category
  handling — all run through the app's real startup lifecycle.

### Week 4 — Dashboard, deployment & docs
- Streamlit dashboard (`dashboard/app.py`), six sections: Overview, Live
  Session Scoring (SHAP-explained, matches the API's output exactly), Funnel
  & Cohort Analytics, Model Comparison, Explainability, Performance Metrics.
  Reads saved artifacts and exported CSVs only — never retrains.
- Dockerfiles for both the API and dashboard, each with its own lean,
  service-specific `requirements-*.txt` (not the full 150-package dev
  environment) and a non-root user.
- `docker-compose.yml` brings both services up together with one command.
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff lint → black format
  check → pytest → Docker builds for both images, in that order, on every
  push to `main`.
- MLflow experiment tracking (`mlflow.db`, not committed — regenerable):
  all six baseline models, the imbalance-strategy benchmark, and both
  hyperparameter-tuning runs logged with params and metrics; the final
  CatBoost model registered in the MLflow Model Registry as
  `purchase-intent-catboost`.
- AWS deployment guide (`docs/aws_deployment.md`) — ECR/ECS steps documented;
  no live AWS resources provisioned for this submission.

---

## Running the notebooks

```bash
jupyter lab
```

Open notebooks in order: `01_eda.ipynb` → `02_features.ipynb` →
`03_modeling.ipynb` → `04_xai.ipynb`. Each restarts cleanly and runs top to
bottom without errors.

---

## Running the API

```bash
uvicorn src.api.main:app --reload
```

Visit http://localhost:8000/docs for interactive Swagger docs, or:

```bash
curl http://localhost:8000/health
```

---

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at http://localhost:8501.

---

## Running everything with Docker Compose

```bash
docker-compose up --build
```

Brings up both containers together. API on port 8000, dashboard on port 8501.
Tear down with:

```bash
docker-compose down
```

---

## Running tests

```bash
pytest tests/ -v
```

(`pytest.ini` sets `pythonpath = .` so this works from a fresh clone without
extra setup.)

---

## MLflow experiment tracking

```bash
mlflow ui
```

Opens at http://localhost:5000, "Model training" tab. Shows every model
comparison, strategy benchmark, and tuning run logged during Week 2, plus the
registered `purchase-intent-catboost` model under "Model registry."

---

## CI/CD

Every push to `main` triggers `.github/workflows/ci.yml`: ruff → black
`--check` → pytest → Docker build for both images. See the **Actions** tab on
GitHub for run history.

---

## AWS deployment

See `docs/aws_deployment.md` for the documented ECR + ECS Fargate deployment
path. Not provisioned for this submission (no live cloud spend required).

---

## Documentation

- [`docs/model_card.md`](docs/model_card.md) — model details, performance,
  explainability summary, limitations, ethical considerations
- [`docs/aws_deployment.md`](docs/aws_deployment.md) — AWS deployment guide
- SHAP/LIME visualizations: `docs/shap_*.png`, `docs/lime_*.png`

---

## Known limitations

See `docs/model_card.md` for the full list. Headline items:
- `PageValues` is the model's strongest feature but carries a leakage risk
  (it reflects pages seen close to checkout); kept in the model and
  documented rather than dropped.
- The `$50`/`$5` figures behind the 0.45 decision threshold are illustrative,
  not supplied by a real retailer.
- The `Other` visitor-type segment has only 133 sessions in the dataset —
  conclusions about it should be treated with lower confidence.
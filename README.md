# Rossmann Sales Prediction Pipeline

![GitHub Repo stars](https://img.shields.io/github/stars/ksnishat/rossmann-sales-pipeline?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/ksnishat/rossmann-sales-pipeline)
![Python](https://img.shields.io/badge/Python-3.10-00599C?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-3F4F75?style=flat&logo=plotly)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0171B9?style=flat&logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/K8s-Deployment-326CE5?style=flat&logo=kubernetes)
![Helm](https://img.shields.io/badge/Helm-Charts-0F1689?style=flat&logo=helm)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=flat&logo=githubactions)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![OS: Linux](https://img.shields.io/badge/OS-Linux-2F2F2F?style=flat)

---

An **end-to-end Machine Learning solution** to predict **daily sales** for Rossmann drug stores. This project implements a complete **MLOps pipeline**, including data extraction, experiment tracking, model deployment via REST API, and a professional multi-page dashboard built with **Plotly Dash** (modern alternative to Streamlit for German enterprise market).

## Key Features

- **Machine Learning:** Random Forest Regressor trained on store, promotion, and competitor data.
- **Experiment Tracking:** Uses **MLflow** to track model metrics and parameters.
- **Backend API:** Served using **FastAPI** for real-time predictions with health checks and Prometheus metrics.
- **Multi-page Dashboard:** Interactive UI built with **Plotly Dash** and **Dash Bootstrap Components** for enterprise-grade visualization.
- **Containerization:** Fully Dockerized for easy deployment with **Docker Compose**.
- **Database:** **PostgreSQL** integration for raw data storage and feature store.
- **Production Ready:** Kubernetes and Helm charts for scalable deployment.
- **CI/CD Pipeline:** GitHub Actions for automated testing, building, and deployment.


### Recent Improvements (2026)
🔧 **Makefile** — Standardized commands: `make test`, `make lint`, `make docker-up`, `make k8s-deploy`, `make promote`
📦 **pyproject.toml** — Modern Python packaging with dependencies, entry points, ruff/mypy config
🔒 **Pre-commit hooks** — Ruff, mypy, black, trailing whitespace, YAML validation
✅ **Data Quality Validation** — Schema-based validation (completeness, validity, consistency, uniqueness)
☁️ **Terraform IaC** — Azure infrastructure as code (AKS, PostgreSQL, Redis, monitoring)
📊 **Prometheus instrumentation** — the API now exposes request, latency, prediction and model-loaded metrics
🐛 **Pydantic v2 fix** — `data.dict()` → `data.model_dump()`
🐛 **Compose fix** — removed stale `ghcr.io/rossmann/*` image references so `dash` and `streamlit` build locally
✅ **Test suite green** — 25/25 passing

## Architecture

```mermaid
flowchart LR
    RAW["Raw sales<br/>CSV + PostgreSQL"] --> FEAT["Feature engineering<br/>holidays, promos, competition"]
    FEAT --> STORE["Feature store<br/>train_cleaned"]
    STORE --> RF["Random Forest<br/>scikit-learn"]
    RF --> MLflow["MLflow<br/>runs + metrics"]
    MLflow --> PROMOTE["Promotion<br/>best run by MAE"]
    PROMOTE --> API["FastAPI<br/>/predict"]
    API --> DASH["Plotly Dash<br/>4-page BI dashboard"]
    API --> Prom["Prometheus"]
    Prom --> Graf["Grafana"]

    classDef data fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#0d47a1
    classDef train fill:#fff3e0,stroke:#e65100,stroke-width:1px,color:#bf360c
    classDef track fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px,color:#4a148c
    classDef serve fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#1b5e20
    classDef observe fill:#eceff1,stroke:#37474f,stroke-width:1px,color:#263238

    class RAW,FEAT,STORE data
    class RF train
    class MLflow,PROMOTE track
    class API,DASH serve
    class Prom,Graf observe
```

## Professional Feature Table

| Feature | Description | Business Value | German Industry Relevance |
|---------|-------------|----------------|---------------------------|
| **Sales Forecasting** | Random Forest for daily sales prediction | Improves inventory accuracy | Supports German retail planning (Einzelhandel) standards |
| **Multi-page Dash Dashboard** | Professional Plotly Dash with 4 pages: Home, Stores, Predict, Metrics | Enables data-driven store management decisions | Matches German retail dashboard requirements (KPI-Cockpit) |
| **MLflow Experiment Tracking** | Full experiment lineage with parameters, metrics, artifacts | Ensures model reproducibility and auditability | Complies with German regulatory audit requirements (GoBD) |
| **PostgreSQL Feature Store** | Centralized feature storage with versioning | Reduces feature engineering duplication | Aligns with German data governance standards |
| **Prometheus Monitoring** | Custom metrics for prediction latency, request counts and last predicted value | Proactive service monitoring; no accuracy or drift metric is emitted because ground-truth sales are not available at serving time | Supports DIN SPEC 92001 AI quality standards |
| **CI/CD Pipeline** | Automated testing, building, deployment | Reduces deployment risk and time-to-market | DevOps practices valued by German enterprises |
| **Multi-language Support** | Dashboard available in German/English | Facilitates adoption across German retail chains | Meets German language requirements in B2B software |
| **Scalable Architecture** | Kubernetes HPA with CPU/memory scaling | Handles Black Friday / peak season loads | Critical for German retail peak periods (Weihnachtsgeschäft) |

## Why This Matters for German Industry

Germany's retail sector is undergoing digital transformation, with predictive analytics becoming a competitive differentiator:

1. **Retail 4.0 Initiative**: The platform aligns with HDE (Handelsverband Deutschland) digitalization roadmap for retail analytics.

2. **Inventory Optimization**: Accurate sales forecasts reduce overstock (Verpackungsgesetz compliance) and understock (customer satisfaction).

3. **Workforce Planning**: Sales forecasts enable optimal shift scheduling under German Arbeitszeitgesetz (working time law).

4. **Promotion Effectiveness**: Quantifies promotion ROI, critical for German drugstore chains (dm, Rossmann, Müller) competing on promotions.

5. **Data Sovereignty**: On-premise/Kubernetes deployment keeps sensitive sales data within German borders (GDPR/BDSG).

6. **SME Accessibility**: Helm charts and Docker Compose enable Mittelstand retailers to deploy without cloud vendor lock-in.

## Verified Model Metrics

Reproduced locally on an NVIDIA RTX 3050 Ti (4 GB).

| Metric | Value |
|--------|-------|
| Dataset | [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle) |
| Training rows | 844,392 |
| Model | `RandomForestRegressor` (scikit-learn) |
| **MAE** | **869.51** |
| **RMSE** | **1295.21** |
| Target | `log1p(Sales)`, inverted with `expm1` at inference |

The Kaggle competition's winning solutions reach ~0.10 RMSPE using gradient
boosting plus extensive feature engineering (store-level history, holiday
proximity, promo decay). This baseline is a clean, reproducible reference
implementation — the value here is the *pipeline*, not the leaderboard rank.

## Quickstart

### 1. Install dependencies

```bash
conda create -n rossmann-sales-env python=3.10 -y
conda activate rossmann-sales-env
pip install -r requirements.txt
```

### 2. Get the data

Download `train.csv` and `store.csv` from the
[Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales/data) and
place them in `rossmann-store-sales/` at the repository root.

### 3. Train

```bash
python -m src.scripts.train_model
```

This writes `models/model.pkl` and logs the run to MLflow.

### 4. Start the API

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8002
```

### 5. Verify

```bash
# The root path is the health endpoint; there is no /health route.
curl http://localhost:8002/
# {"status":"healthy","service":"rossmann-sales-api","model_loaded":true}

curl http://localhost:8002/metrics | grep rossmann
# Field names are PascalCase and match the training columns exactly.
curl -X POST http://localhost:8002/predict \
     -H 'Content-Type: application/json' \
     -d '{"Store":1,"DayOfWeek":5,"Promo":1,"SchoolHoliday":0,"StoreType":"a","Assortment":"a","CompetitionDistance":1270.0,"Promo2":0}'
# {"predicted_sales":5034.717003288615}
```

### 6. Or launch the whole Docker stack

```bash
docker compose up -d      # postgres, mlflow, api, dash, streamlit, prometheus, grafana
```

## Running Tests

```bash
pytest tests/ -v          # 34 passed, 8 skipped
```

The 8 skipped tests are integration tests that need the live docker stack. They
are excluded from CI (which has no model checkpoint or dataset) and run against
real services instead of an in-process `TestClient`:

```bash
./run_stack.sh
LIVE_STACK=1 pytest tests/test_integration_live.py -v   # 8 passed
```

They assert the API returns a positive forecast, that a missing required field
is rejected with 422, that Prometheus is actually scraping the API (not merely
healthy), and that MLflow, Grafana and MinIO are reachable.

## Monitoring & Live Demo

```bash
./run_stack.sh                   # API + MLflow + MinIO + Prometheus + Grafana
python3 provision_dashboards.py rossmann   # datasource + dashboard
./verify_stack.sh                 # 7 checks, exit 0 when all pass
```

| Service | URL | Credentials |
|---------|-----|-------------|
| **FastAPI** | http://localhost:8002/docs | N/A |
| **Prometheus** | http://localhost:9092 | N/A |
| **Grafana** | http://localhost:3003 | `admin` / `admin` |
| **Plotly Dash** | http://localhost:8050 | N/A |
| **Streamlit** | http://localhost:8501 | N/A |
| **MLflow** | http://localhost:5003 | N/A |
| **PostgreSQL** | localhost:5433 | rossmann / changeme123 |

### Exposed metrics

| Metric | Type | Meaning |
|--------|------|---------|
| `rossmann_requests_total` | counter | Requests, labelled by `endpoint` and `status` |
| `rossmann_prediction_latency_seconds` | histogram | Prediction latency |
| `rossmann_last_predicted_sales` | gauge | Most recent predicted sales value |
| `rossmann_model_loaded` | gauge | 1 when the model is loaded, 0 otherwise |
| `rossmann_predictions_total` | counter | Predictions, labelled by `source` (`model` / `fallback`) |
| `rossmann_feature_drift_psi` | gauge | PSI per input feature vs the training baseline |
| `rossmann_max_drift_psi` | gauge | Largest per-feature PSI in the current window |

### Drift detection

`GET /drift` reports the Population Stability Index between live request values
and the distribution the model was fitted on. The baseline is captured during
training into `models/drift_baseline.json`.

Only numeric features are tracked: PSI is defined on a continuous or ordinal
distribution, and the one-hot indicator columns are already represented by the
numeric features they derive from. Constant features are dropped because they
collapse the quantile bins and make PSI undefined.

### Scheduled retraining

`dags/retraining_pipeline.py` runs weekly: retrain → promote the best MLflow run
→ reload the serving model → report drift. Promotion is a separate step from
training so a bad run cannot reach production just by finishing.

```bash
docker compose up -d postgres airflow-postgres airflow-webserver airflow-scheduler
docker exec rossmann_airflow_scheduler airflow dags unpause rossmann_continuous_learning
docker exec rossmann_airflow_scheduler airflow dags trigger rossmann_continuous_learning
```

Airflow UI: http://localhost:8083 (`admin` / `admin`).

The reload step calls `POST /admin/reload-model`, disabled unless
`ENABLE_MODEL_RELOAD=true`.

### Building the feature store

`train_model.py` reads `SELECT * FROM train_cleaned`, which nothing created
before. Build it after loading the raw tables:

```bash
python src/scripts/load_data.py           # train + store tables
python src/scripts/build_feature_store.py # train_cleaned
```

## Local Development Setup

### 1. Environment Setup

```bash
conda env create -f environments/rossmann-env.yml
conda activate rossmann-env
```

### 2. Database Setup

```bash
docker compose up -d postgres
python -m src.scripts.init_db
```

### 3. Run API and Dashboards

```bash
# Terminal 1: FastAPI
uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8002

# Terminal 2: Plotly Dash
python -m src.dash_app.app

# Terminal 3: Streamlit (legacy)
streamlit run src/app/dashboard.py
```

## Project Structure

```plaintext
rossmann-sales-pipeline/
├── models/                     # Serialized models (.pkl)
├── notebooks/                  # Jupyter notebooks for EDA
├── src/
│   ├── app/
│   │   ├── api.py              # FastAPI backend + Prometheus instrumentation
│   │   ├── dashboard.py        # Streamlit frontend (legacy)
│   │   └── config.py           # Pydantic settings
│   ├── dash_app/               # Plotly Dash multi-page app
│   │   ├── app.py              # Dash entry point
│   │   ├── layouts.py          # Page layouts
│   │   └── utils.py            # Helper functions
│   └── scripts/
│       ├── train_model.py      # Training script
│       └── init_db.py          # Database initialization
├── tests/                      # 34 unit + 8 integration tests
├── run_stack.sh                # Start API + MLflow + MinIO + Prometheus + Grafana
├── verify_stack.sh             # 7 live-stack health checks
├── k8s/                        # Kubernetes manifests
├── helm-chart/                 # Helm chart for K8s
├── environments/               # Conda environments
├── .github/workflows/          # CI/CD pipelines
└── docker-compose.yml          # Container orchestration
```

## Dashboard Features (Plotly Dash)

The **Plotly Dash** dashboard includes 4 pages:

1. **Home** — Executive KPI overview with sales trends, store count, average sales
2. **Stores** — Interactive store comparison with filters, heatmap, and performance ranking
3. **Predict** — Sales prediction form with feature inputs and forecast visualization
4. **Metrics** — Model performance tracking with MLflow integration

Built with **Dash Bootstrap Components** for a responsive UI.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Database connection failed** | Check PostgreSQL container is running: `docker compose ps postgres` |
| **Model not loaded in API** | Verify `models/model.pkl` exists; the API falls back to a heuristic and reports `source="fallback"` in metrics |
| **`AttributeError: 'X' object has no attribute 'dict'`** | Fixed in this commit — Pydantic v2 uses `model_dump()` |
| **Dash dashboard not loading** | Check port 8050 is not in use; verify `src/dash_app/app.py` imports |
| **MLflow UI not accessible** | Ensure the MLflow container has the correct artifact store configuration |
| **Prediction errors** | Check the input feature schema matches the training features exactly |
| **Grafana shows "No data"** | Re-run `provision_dashboards.py` so the datasource points at the Prometheus container IP |
| **Pods crash on K8s** | Increase resource limits in `helm-chart/values.yaml`; check PVC binding |
| **GitHub Actions failing** | Verify secrets (`GITHUB_TOKEN`, `KUBE_CONFIG_DATA`) are configured |

## Author

Developed by **Khaled Saifullah**.

For collaboration, feature requests, or bug reports, please open an issue or contact the maintainer via the repository issue tracker.

**Last Updated**: October 2026
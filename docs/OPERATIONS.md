# Rossmann — Operations Reference

Setup, configuration and troubleshooting. The [README](../README.md) covers what the project is and how it scores.

## Running Tests
```bash
uv run --frozen --extra dev python -m pytest tests/ -v   # 129 passed, 8 deselected
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

### 1. Environment Setup
```bash
conda env create -f environments/rossmann-env.yml
conda activate rossmann-sales-env
```

### 2. Database Setup
```bash
docker compose up -d postgres
python -m src.scripts.build_feature_store.py
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

## Dashboard Features (Plotly Dash)
The **Plotly Dash** dashboard includes 4 pages:

1. **Home** — Executive KPI overview with sales trends, store count, average sales
2. **Stores** — Interactive store comparison with filters, heatmap, and performance ranking
3. **Predict** — Sales prediction form with feature inputs and forecast visualization
4. **Metrics** — Model performance tracking with MLflow integration

Built with **Dash Bootstrap Components** for a responsive UI.

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
| **Plotly Dash** | http://localhost:8503 | N/A |
| **MLflow** | http://localhost:5003 | N/A |
| **PostgreSQL** | localhost:5433 | rossmann / changeme123 |

> **All four stacks at once:** the root `start_all_stacks.sh` starts the API,
> MLflow, MinIO, Prometheus, Grafana, TLS proxy and this dashboard for every
> project, then `verify_live_stack.sh` checks 32 endpoints. The ports in the
> table above are the ones that script uses.

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
│       └── promote_model.py    # Promote the best MLflow run by MAE
├── tests/                      # 129 unit + 8 integration tests
├── run_stack.sh                # Start API + MLflow + MinIO + Prometheus + Grafana
├── verify_stack.sh             # 7 live-stack health checks
├── k8s/                        # Kubernetes manifests
├── helm-chart/                 # Helm chart for K8s
├── environments/               # Conda environments
├── .github/workflows/          # CI/CD pipelines
└── docker-compose.yml          # Container orchestration
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| **Database connection failed** | Check PostgreSQL container is running: `docker compose ps postgres` |
| **Model not loaded in API** | Verify `models/model.pkl` exists; the API falls back to a heuristic and reports `source="fallback"` in metrics |
| **`AttributeError: 'X' object has no attribute 'dict'`** | Fixed in this commit — Pydantic v2 uses `model_dump()` |
| **Dash dashboard not loading** | Check port 8050 is not in use; verify `src/dash_app/app.py` imports |
| **MLflow UI not accessible** | Ensure the MLflow container has the correct artifact store configuration |
| **Prediction errors** | Check the input feature schema matches the training features exactly |
| **Grafana shows "No data"** | The provisioning or dashboards directory is not mounted, or Grafana is not on the `mlops-stack` network so it cannot resolve `prom_<project>` | Run `python3 provision_dashboards.py` — it reports which of the two it is |
| **Pods crash on K8s** | Increase resource limits in `helm-chart/values.yaml`; check PVC binding |
| **GitHub Actions failing** | Verify secrets (`GITHUB_TOKEN`, `KUBE_CONFIG_DATA`) are configured |

"""
Rossmann Sales Prediction API - FastAPI backend.

Endpoints:
- GET / : Health check
- POST /predict : Predict daily sales for a store
"""

import os
import sys
import time

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth import auth_enabled, require_api_key  # noqa: E402

# Prometheus instrumentation
try:
    from prometheus_client import Counter, Gauge, Histogram

    REQUEST_COUNT = Counter(
        "rossmann_requests_total",
        "Total Rossmann API requests",
        ["endpoint", "status"],
    )
    PREDICTION_LATENCY = Histogram(
        "rossmann_prediction_latency_seconds",
        "Sales prediction latency in seconds",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )
    PREDICTED_SALES = Gauge(
        "rossmann_last_predicted_sales",
        "Most recent predicted daily sales value",
    )
    MODEL_LOADED_GAUGE = Gauge(
        "rossmann_model_loaded",
        "1 if the trained model is loaded, 0 if the fallback heuristic is used",
    )
    MODEL_USED = Counter(
        "rossmann_predictions_total",
        "Predictions served, labelled by whether the trained model was used",
        ["source"],
    )
    DRIFT_PSI_GAUGE = Gauge(
        "rossmann_feature_drift_psi",
        "Population Stability Index per input feature vs the training baseline",
        ["feature"],
    )
    DRIFT_MAX_PSI_GAUGE = Gauge(
        "rossmann_max_drift_psi",
        "Largest per-feature PSI in the current window",
    )
except ImportError:  # pragma: no cover
    REQUEST_COUNT = PREDICTION_LATENCY = PREDICTED_SALES = None
    MODEL_LOADED_GAUGE = MODEL_USED = None
    DRIFT_PSI_GAUGE = DRIFT_MAX_PSI_GAUGE = None

# Global model variables
_model = None
_model_columns = None
_model_loaded = False
_drift_monitor = None


def load_model() -> bool:
    """Load model and columns at startup with graceful fallback."""
    global _model, _model_columns, _model_loaded

    model_path = os.getenv("MODEL_PATH", "models/production_model.pkl")
    columns_path = os.getenv("MODEL_COLUMNS_PATH", "models/model_columns.pkl")

    if os.path.exists(model_path) and os.path.exists(columns_path):
        try:
            _model = joblib.load(model_path)
            _model_columns = joblib.load(columns_path)
            _model_loaded = True
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")

    print("No model found - using fallback prediction")
    return False


def load_drift_monitor() -> bool:
    """Load the training-time feature baseline used for PSI drift detection.

    Best-effort: without a baseline the API still serves predictions, it just
    cannot report drift.
    """
    global _drift_monitor

    try:
        from .drift import DriftMonitor

        path = os.getenv("DRIFT_BASELINE_PATH", "models/drift_baseline.json")
        monitor = DriftMonitor.from_file(path)
        if not monitor.enabled:
            print(f"No drift baseline at {path}; drift metrics disabled")
            _drift_monitor = None
            return False
        _drift_monitor = monitor
        print(f"Drift baseline loaded ({len(monitor.baseline)} features)")
        return True
    except Exception as e:  # noqa: BLE001 - drift reporting is optional
        print(f"Drift baseline unavailable ({e})")
        _drift_monitor = None
        return False


app = FastAPI(title="Rossmann Sales Prediction API")


# Load model at startup
load_model()
load_drift_monitor()


class SalesInput(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Promo2: int


def _predict_sales(model, columns, input_data: dict) -> float:
    """Internal prediction logic."""
    df = pd.DataFrame([input_data])
    df['Open'] = 1
    df_processed = pd.get_dummies(df)
    df_final = df_processed.reindex(columns=columns, fill_value=0)
    prediction = model.predict(df_final)
    return float(prediction[0])


def _fallback_prediction(input_data: dict) -> float:
    """Simple fallback prediction when model is not available."""
    # Simple heuristic based on inputs
    base = 5000
    base += input_data["Promo"] * 800
    base += (input_data["DayOfWeek"] in [6, 7]) * 1500  # Weekend boost
    base += input_data["SchoolHoliday"] * 500
    base -= min(input_data["CompetitionDistance"] / 1000 * 200, 500)
    return max(0, base)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "rossmann-sales-api",
        "model_loaded": _model_loaded,
        "auth_enabled": auth_enabled(),
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict_sales(data: SalesInput):
    """
    Predict daily sales for a Rossmann store.

    Returns:
        JSON with predicted_sales (float)
    """
    try:
        input_data = data.model_dump()

        t0 = time.perf_counter()
        if _model_loaded and _model is not None:
            try:
                prediction = _predict_sales(_model, _model_columns, input_data)
                source = "model"
            except Exception as e:
                print(f"Model prediction failed: {e}")
                prediction = _fallback_prediction(input_data)
                source = "fallback"
        else:
            prediction = _fallback_prediction(input_data)
            source = "fallback"

        # Record metrics
        if REQUEST_COUNT is not None:
            PREDICTION_LATENCY.observe(time.perf_counter() - t0)
            PREDICTED_SALES.set(prediction)
            MODEL_USED.labels(source=source).inc()
            REQUEST_COUNT.labels(endpoint="predict", status="200").inc()

        drift = observe_drift(input_data)
        if DRIFT_MAX_PSI_GAUGE is not None and drift.get("features"):
            for feature, stats in drift["features"].items():
                DRIFT_PSI_GAUGE.labels(feature=feature).set(stats["psi"])
            DRIFT_MAX_PSI_GAUGE.set(max(s["psi"] for s in drift["features"].values()))

        return {"predicted_sales": prediction, "drift": drift}

    except Exception as e:
        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(endpoint="predict", status="500").inc()
        return {"error": str(e)}


def observe_drift(input_data: dict) -> dict:
    """Feed the request features into the drift monitor.

    Never raises: drift reporting must not break a prediction.
    """
    if _drift_monitor is None or not _drift_monitor.enabled:
        return {"enabled": False}

    try:
        _drift_monitor.observe({k: v for k, v in input_data.items() if isinstance(v, (int, float))})
        return {"enabled": True, "recorded": True, "features": _drift_monitor.compute()}
    except Exception as exc:  # noqa: BLE001 - drift must never break /predict
        return {"enabled": True, "recorded": False, "error": str(exc)}


@app.get("/drift", dependencies=[Depends(require_api_key)])
def drift_report():
    """Per-feature PSI against the training baseline."""
    if _drift_monitor is None or not _drift_monitor.enabled:
        return {
            "enabled": False,
            "detail": "No drift baseline loaded. Set DRIFT_BASELINE_PATH to enable.",
        }

    features = _drift_monitor.compute()
    if not features:
        return {
            "enabled": True,
            "status": "insufficient_data",
            "detail": "Fewer than the minimum samples observed so far.",
            "features": {},
        }

    worst = max(features.items(), key=lambda kv: kv[1]["psi"])
    return {
        "enabled": True,
        "status": worst[1]["status"],
        "max_psi": worst[1]["psi"],
        "worst_feature": worst[0],
        "features": features,
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    if MODEL_LOADED_GAUGE is not None:
        MODEL_LOADED_GAUGE.set(1 if _model_loaded else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/reload-model", dependencies=[Depends(require_api_key)])
def reload_model():
    """Re-read the promoted model from disk without restarting the API.

    The retraining DAG promotes the best MLflow run, then calls this so the
    fresh model is served immediately. Disabled unless ENABLE_MODEL_RELOAD=true:
    swapping model weights is a privileged operation and should not be reachable
    by default.
    """
    if os.getenv("ENABLE_MODEL_RELOAD", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=403,
            detail="Model reload is disabled. Set ENABLE_MODEL_RELOAD=true to enable it.",
        )

    if not load_model():
        raise HTTPException(status_code=500, detail="Reload failed: model could not be loaded")

    return {
        "status": "reloaded",
        "model_loaded": _model_loaded,
        "model_path": os.getenv("MODEL_PATH", "models/production_model.pkl"),
    }

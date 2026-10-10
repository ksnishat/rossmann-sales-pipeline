"""
Rossmann Sales Prediction API - FastAPI backend.

Endpoints:
- GET / : Health check
- POST /predict : Predict daily sales for a store
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time
from typing import Optional

# Prometheus instrumentation
try:
    from prometheus_client import Counter, Histogram, Gauge

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
except ImportError:  # pragma: no cover
    REQUEST_COUNT = PREDICTION_LATENCY = PREDICTED_SALES = None
    MODEL_LOADED_GAUGE = MODEL_USED = None

# Global model variables
_model = None
_model_columns = None
_model_loaded = False


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


app = FastAPI(title="Rossmann Sales Prediction API")


# Load model at startup
load_model()


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
    }


@app.post("/predict")
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

        return {"predicted_sales": prediction}

    except Exception as e:
        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(endpoint="predict", status="500").inc()
        return {"error": str(e)}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    if MODEL_LOADED_GAUGE is not None:
        MODEL_LOADED_GAUGE.set(1 if _model_loaded else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

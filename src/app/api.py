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
from typing import Optional

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
        input_data = data.dict()

        if _model_loaded and _model is not None:
            try:
                prediction = _predict_sales(_model, _model_columns, input_data)
                return {"predicted_sales": prediction}
            except Exception as e:
                print(f"Model prediction failed: {e}")

        # Fallback
        prediction = _fallback_prediction(input_data)
        return {"predicted_sales": prediction}

    except Exception as e:
        return {"error": str(e)}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
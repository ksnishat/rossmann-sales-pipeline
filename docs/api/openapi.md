# Rossmann Sales Prediction API Documentation

> OpenAPI 3.0 Specification for Rossmann Sales Prediction Backend API

## Base URL

```
http://localhost:8000
```

## Endpoints

### `GET /`

Root endpoint.

**Response:**
```json
{
  "service": "Rossmann Sales Prediction API",
  "version": "1.0.0",
  "description": "Retail Sales Forecasting API"
}
```

---

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-10-02T10:30:00.000Z"
}
```

---

### `POST /predict`

Predict daily sales for a given store.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | integer | Yes | Store identifier (1-1115) |
| `date` | string | Yes | Date (YYYY-MM-DD) |
| `is_promo` | boolean | No | Whether promotion is active |
| `is_holiday` | boolean | No | Whether school/holiday |
| `is_weekend` | boolean | No | Whether weekend |
| `assortment` | string | No | Assortment level (a/b/c) |
| `store_type` | string | No | Store type (a/b/c/d) |
| `competition_distance` | float | No | Distance to nearest competitor (meters) |
| `competition_open_since` | string | No | Competitor open since date |
| `promo_since` | string | No | Promo start date |
| `promo_interval` | string | No | Promo interval description |

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `store_id` | integer | Store identifier |
| `date` | string | Prediction date |
| `predicted_sales` | float | Predicted daily sales |
| `predicted_customers` | integer | Estimated customers |
| `confidence_interval` | array[float] | 95% confidence interval [low, high] |
| `processing_time_ms` | float | Processing time in ms |
| `model_version` | string | Model version used |
| `timestamp` | string | ISO format timestamp |

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 45,
    "date": "2026-11-01",
    "is_promo": true,
    "is_holiday": false,
    "is_weekend": true,
    "assortment": "c",
    "store_type": "b",
    "competition_distance": 1234.5
  }'
```

**Example Response:**
```json
{
  "store_id": 45,
  "date": "2026-11-01",
  "predicted_sales": 7850.42,
  "predicted_customers": 542,
  "confidence_interval": [7200.5, 8500.3],
  "processing_time_ms": 5.2,
  "model_version": "random_forest_v1.2.0",
  "timestamp": "2026-10-02T10:30:00.000Z"
}
```

---

### `POST /predict/batch`

Batch prediction for multiple stores.

**Request Body:**
Array of prediction request objects (same schema as `/predict`)

**Response:** Array of prediction responses.

---

### `GET /models`

List available models.

**Response:**
```json
[
  {"name": "rossmann-sales-model", "version": "1.0.0", "stage": "Production"},
  {"name": "rossmann-sales-model", "version": "1.1.0", "stage": "Staging"}
]
```

---

### `GET /docs` and `GET /openapi.json`

Interactive API docs and OpenAPI spec.

## Metrics

- `rossmann_prediction_seconds` - Prediction latency histogram
- `rossmann_predictions_total` - Total predictions counter
- `rossmann_model_accuracy` - Current model accuracy gauge
- `rossmann_api_requests_total` - Total API request counter
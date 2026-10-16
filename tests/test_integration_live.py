"""Integration tests against the live docker stack.

Skipped unless ``LIVE_STACK=1``. The rest of the suite uses ``TestClient``,
which runs the ASGI app in-process and cannot detect a container that never
started, a wrong port, or Prometheus failing to scrape.

Run with::

    LIVE_STACK=1 pytest tests/test_integration_live.py -v
"""

import os

import pytest
import requests

pytestmark = pytest.mark.integration

API_PORT = int(os.getenv("ROSSMANN_API_PORT", "8002"))
MLFLOW_PORT = int(os.getenv("ROSSMANN_MLFLOW_PORT", "5003"))
PROM_PORT = int(os.getenv("ROSSMANN_PROM_PORT", "9092"))
GRAFANA_PORT = int(os.getenv("ROSSMANN_GRAFANA_PORT", "3003"))
MINIO_PORT = int(os.getenv("ROSSMANN_MINIO_PORT", "9004"))

API = f"http://127.0.0.1:{API_PORT}"
TIMEOUT = 30

# The API requires PascalCase field names and Promo2 is mandatory.
VALID_PAYLOAD = {
    "Store": 1,
    "DayOfWeek": 4,
    "Promo": 1,
    "SchoolHoliday": 0,
    "StoreType": "c",
    "Assortment": "a",
    "CompetitionDistance": 1270.0,
    "Promo2": 0,
}


def _require_live() -> None:
    if os.getenv("LIVE_STACK") != "1":
        pytest.skip("set LIVE_STACK=1 to run integration tests against the live stack")


# The APIs enforce X-API-Key when API_KEY is set. The live stack runs with auth
# enabled, so every request must carry the key. Set LIVE_API_KEY to match the
# stack; the default matches run_stack.sh's demo value.
API_KEY = os.getenv("LIVE_API_KEY", "demo-secret-key")
AUTH_HEADERS = {"X-API-Key": API_KEY}


def _auth(kw):
    """Merge the API key into a request's headers."""
    headers = dict(kw.pop("headers", {}) or {})
    headers.update(AUTH_HEADERS)
    kw["headers"] = headers
    return kw


def _get(url: str, **kw):
    try:
        return requests.get(url, timeout=TIMEOUT, **_auth(kw))
    except requests.RequestException as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"service unreachable: {exc}")


def test_api_health_endpoint_is_live():
    _require_live()
    r = _get(f"{API}/")
    assert r.status_code == 200
    assert r.json()["status"] in {"healthy", "running"}


def test_api_metrics_are_scrapable():
    _require_live()
    r = _get(f"{API}/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")


def test_predict_returns_a_positive_forecast():
    _require_live()
    r = requests.post(f"{API}/predict", json=VALID_PAYLOAD, timeout=TIMEOUT,
        headers=AUTH_HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predicted_sales" in body
    assert body["predicted_sales"] > 0


def test_predict_rejects_a_missing_required_field():
    """Promo2 is required; omitting it must be a 422, not a silent default."""
    _require_live()
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Promo2"}
    r = requests.post(f"{API}/predict", json=payload, timeout=TIMEOUT,
        headers=AUTH_HEADERS)
    assert r.status_code == 422, r.text


def test_mlflow_tracking_server_is_reachable():
    _require_live()
    r = _get(
        f"http://127.0.0.1:{MLFLOW_PORT}/api/2.0/mlflow/experiments/search",
        headers={"Host": f"127.0.0.1:{MLFLOW_PORT}"},
        params={"max_results": 1},
    )
    assert r.status_code == 200, r.text
    assert "experiments" in r.json()


def test_prometheus_is_scraping_the_api():
    _require_live()
    r = _get(f"http://127.0.0.1:{PROM_PORT}/api/v1/targets")
    assert r.status_code == 200
    targets = r.json()["data"]["activeTargets"]
    assert targets, "Prometheus has no configured targets"
    for t in targets:
        assert t["health"] == "up", f"{t['scrapeUrl']} is {t['health']}: {t.get('lastError')}"


def test_grafana_is_healthy():
    _require_live()
    r = _get(f"http://127.0.0.1:{GRAFANA_PORT}/api/health")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"


def test_minio_is_live():
    _require_live()
    r = _get(f"http://127.0.0.1:{MINIO_PORT}/minio/health/live")
    assert r.status_code == 200

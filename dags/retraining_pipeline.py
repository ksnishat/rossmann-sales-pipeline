# FILE: dags/retraining_pipeline.py
"""Scheduled retraining for the Rossmann sales forecaster.

Flow: retrain -> promote the best run to production -> reload the serving model
-> report drift.

Every project import is inside the task callables. A module-level import of
sklearn or sqlalchemy means a dependency problem makes the DAG file fail to
parse, and the DAG then silently disappears from the Airflow UI instead of
reporting an error.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to path so Airflow can find the project modules. This must point
# outside /opt/airflow/dags: Airflow scans that directory for DAG files, so
# mounting src inside it makes Airflow try to parse every project module as a
# DAG and report import errors.
sys.path.append(os.getenv("ROSSMANN_SRC", "/opt/airflow/src"))


def retrain_model(**context):
    """Retrain the forecaster and return a JSON-serializable summary for XCom."""
    from scripts.train_model import train

    train(
        n_estimators=int(os.getenv("ROSSMANN_N_ESTIMATORS", "50")),
        max_depth=int(os.getenv("ROSSMANN_MAX_DEPTH", "10")),
    )
    return {"status": "trained", "ds": context["ds"]}


def promote_model(**context):
    """Copy the best MLflow run's model into the serving location.

    Training logs every run to MLflow; promotion is what decides which one is
    actually served. Keeping them separate means a bad run cannot reach
    production just by finishing.
    """
    from scripts.promote_model import promote_best_model

    promote_best_model()
    return {"status": "promoted"}


def reload_serving_model(**context):
    """Tell the serving API to pick up the promoted model.

    Best-effort by design: if the API is not running (for example when only the
    DAG is being exercised) the task logs that and succeeds. The point is to
    make the retrain -> serve handoff visible in the DAG graph and to fail
    loudly when the API is reachable but rejects the new weights.
    """
    import json
    import urllib.error
    import urllib.request

    url = os.getenv(
        "ROSSMANN_API_RELOAD_URL", "http://host.docker.internal:8002/admin/reload-model"
    )

    try:
        request = urllib.request.Request(url, method="POST", data=b"")
        # The serving API enforces X-API-Key on /admin/reload-model. Without
        # this header the call returns 401, which is not in the skip list
        # below, so the task raised and the whole DAG failed.
        api_key = os.getenv("API_KEY")
        if api_key:
            request.add_header("X-API-Key", api_key)
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
        return {"status": "reloaded", "response": payload}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:200]
        if exc.code in (403, 404):
            # Reload is disabled or not implemented, not broken.
            return {"status": "skipped", "reason": f"HTTP {exc.code}: {detail}"}
        raise RuntimeError(f"API rejected the reload ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        return {"status": "skipped", "reason": f"API unreachable: {exc.reason}"}


def report_drift(**context):
    """Read the serving API's drift state so the run records it."""
    import json
    import urllib.error
    import urllib.request

    url = os.getenv("ROSSMANN_DRIFT_URL", "http://host.docker.internal:8002/drift")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode())
        return {"status": "reported", "drift": payload}
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        return {"status": "skipped", "reason": str(exc)}


default_args = {
    "owner": "nishat",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="rossmann_continuous_learning",
    default_args=default_args,
    description="Weekly retraining + promotion + model reload + drift report",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["rossmann", "mlops"],
) as dag:
    train_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    promote_task = PythonOperator(
        task_id="promote_best_model",
        python_callable=promote_model,
    )

    reload_task = PythonOperator(
        task_id="reload_serving_model",
        python_callable=reload_serving_model,
    )

    drift_task = PythonOperator(
        task_id="report_drift",
        python_callable=report_drift,
    )

    train_task >> promote_task >> reload_task >> drift_task

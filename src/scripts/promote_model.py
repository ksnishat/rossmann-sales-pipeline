import os
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn


def promote_best_model(model_dir: Optional[str] = None) -> str:
    """Copy the best MLflow run's model into the serving location.

    Returns the path the model was written to.

    Falls back to the locally trained model when the MLflow run has no logged
    artifacts. Artifact logging is best-effort (the tracking server may use an
    artifact root the training process cannot write to), so a run can exist with
    metrics but no model. Without this fallback promotion would fail on every
    containerised run.
    """
    model_dir = model_dir or os.getenv("ROSSMANN_MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_experiment("rossmann_sales")

    # 1. Find Best Run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("rossmann_sales")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No MLflow runs found; train a model before promoting")

    best_run = runs[0]
    run_id = best_run.info.run_id
    print(f"Promoting Run ID: {run_id} (MAE {best_run.data.metrics.get('mae')})")

    # 2. Load Model, falling back to the local artifact when the run has none
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception as e:  # noqa: BLE001 - the run may have no artifacts
        local_path = os.path.join(model_dir, "candidate_model.pkl")
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"Run {run_id} has no logged model and no local candidate at "
                f"{local_path}: {e}"
            ) from e
        print(f"Run has no logged artifacts ({e}); using {local_path}")
        model = joblib.load(local_path)

    # 3. Save Model
    target = os.path.join(model_dir, "production_model.pkl")
    joblib.dump(model, target)
    print(f"Promoted model written to {target}")

    # 4. Save Feature Names
    # Instead of guessing from SQL, we ask the model what it learned.
    # Random Forest stores the feature names in .feature_names_in_
    try:
        features = model.feature_names_in_.tolist()
        joblib.dump(features, os.path.join(model_dir, "model_columns.pkl"))
        print(f"Schema saved! Model expects {len(features)} columns.")
    except AttributeError:
        print("Warning: This model doesn't have .feature_names_in_. Training might be too old.")

    return target


if __name__ == "__main__":
    promote_best_model()

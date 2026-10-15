import argparse  # NEW: Allows us to pass settings from terminal
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# 1. SETUP EXPERIMENT
mlflow.set_experiment("rossmann_sales")

def train(n_estimators, max_depth):
    print(f"Starting Run: Trees={n_estimators}, Depth={max_depth}")

    # 2. LOAD DATA
    # Configurable so the same code runs on the host (localhost) and inside
    # the Airflow container (the postgres service name).
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://rossmann:changeme123@localhost:5432/rossmann",
    )
    engine = create_engine(db_url)
    df = pd.read_sql("SELECT * FROM train_cleaned", engine)

    # 3. PREPARE
    X = df.drop(columns=['Sales', 'Date', 'Customers', 'StateHoliday'])
    y = df['Sales']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. TRAIN & LOG
    with mlflow.start_run():
        # Log the settings (So we know which combo created this result)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Result: MAE = {mae:.2f}")

        # Log metric
        mlflow.log_metric("mae", mae)

        # Artifact logging is best-effort. The tracking server may use an
        # artifact root this process cannot write to (for example a host path
        # when training runs inside a container), and losing the run's artifacts
        # must not fail an otherwise successful training job. Params and metrics
        # still reach the backend store over HTTP.
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:  # noqa: BLE001 - artifacts are optional
            print(f"MLflow artifact logging skipped ({e})")

        # Always write a local candidate. Promotion prefers the MLflow artifact
        # but falls back to this file, so a run whose artifacts could not be
        # uploaded is still promotable.
        model_dir = os.getenv("ROSSMANN_MODEL_DIR", "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "candidate_model.pkl"))
        print(f"Candidate model written to {model_dir}/candidate_model.pkl")

        # 5. CAPTURE THE DRIFT BASELINE
        # Serving compares incoming requests against the distribution the model
        # was fitted on, so persist the numeric feature distributions here.
        # Without this the API has no baseline and cannot report PSI.
        try:
            _save_drift_baseline(X_train, "models/drift_baseline.json")
            print("Drift baseline saved to models/drift_baseline.json")
        except Exception as e:  # noqa: BLE001 - must not fail an otherwise good run
            print(f"Drift baseline capture failed ({e}); drift metrics will be unavailable")


def _save_drift_baseline(X_train: pd.DataFrame, path: str) -> None:
    """Persist per-feature distributions of the training data for PSI drift.

    Only numeric columns are stored: PSI is defined on a continuous or ordinal
    distribution, and the one-hot indicator columns are already represented by
    the numeric features they were derived from.
    """
    import json
    import os

    numeric = X_train.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("no numeric columns to build a drift baseline from")

    # Cap the sample count so the JSON stays small; PSI needs the shape of the
    # distribution, not every row.
    max_samples = 20000
    if len(numeric) > max_samples:
        numeric = numeric.sample(n=max_samples, random_state=42)

    features = {}
    dropped = []
    for name in numeric.columns:
        column = numeric[name].astype(float)
        if float(column.std()) < 1e-9:
            # A constant feature collapses the quantile bins and makes PSI
            # undefined, so it is dropped rather than reported as drift.
            dropped.append(str(name))
            continue
        features[str(name)] = [round(float(v), 6) for v in column]

    if dropped:
        print(f"Dropped {len(dropped)} constant feature(s) from the drift baseline")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"features": features}, fh)

if __name__ == "__main__":
    # This block handles the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train(args.n_estimators, args.max_depth)

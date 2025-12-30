import mlflow
import mlflow.sklearn
import joblib
import os
import pandas as pd

def promote_best_model():
    mlflow.set_experiment("rossmann_sales")
    
    # 1. Find Best Run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("rossmann_sales")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=1
    )
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    print(f"Promoting Run ID: {run_id}")
    
    # 2. Load Model
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    
    # 3. Create 'models' folder
    os.makedirs("models", exist_ok=True)
    
    # 4. Save Model
    joblib.dump(model, "models/production_model.pkl")
    
    # 5. Save Feature Names (THE FIX)
    # Instead of guessing from SQL, we ask the model what it learned!
    # Random Forest stores the feature names in .feature_names_in_
    try:
        features = model.feature_names_in_.tolist()
        joblib.dump(features, "models/model_columns.pkl")
        print(f"Schema saved! Model expects {len(features)} columns.")
    except AttributeError:
        print("Warning: This model doesn't have .feature_names_in_. Training might be too old.")

if __name__ == "__main__":
    promote_best_model()
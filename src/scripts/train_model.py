import pandas as pd
import argparse  # NEW: Allows us to pass settings from terminal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine

# 1. SETUP EXPERIMENT
mlflow.set_experiment("rossmann_sales")

def train(n_estimators, max_depth):
    print(f"Starting Run: Trees={n_estimators}, Depth={max_depth}")
    
    # 2. LOAD DATA
    db_url = "postgresql://user:password@localhost:5432/rossmann"
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
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    # This block handles the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    train(args.n_estimators, args.max_depth)
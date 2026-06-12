"""
Rossmann Sales DVC Pipeline - Gold Layer
Creates lag features, rolling statistics for time series forecasting.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

LAGS = [1, 2, 3, 7, 14, 28, 30, 90]
ROLLING_WINDOWS = [7, 14, 30, 90]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features from silver data."""
    df = df.copy()
    df = df.sort_values(["Store", "Date"])
    
    # Lag features
    for lag in LAGS:
        df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
    
    # Rolling statistics
    for window in ROLLING_WINDOWS:
        df[f"Sales_rolling_mean_{window}"] = df.groupby("Store")["Sales"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"Sales_rolling_std_{window}"] = df.groupby("Store")["Sales"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Date features
    df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/silver", help="Silver data directory")
    parser.add_argument("--output", default="data/gold", help="Gold data directory")
    args = parser.parse_args()
    
    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    
    # Process train.csv
    train_file = src / "train.csv"
    if train_file.exists():
        df = pd.read_csv(train_file)
        df = create_features(df)
        df.to_csv(dst / "train_features.csv", index=False)
        print(f"Gold: train_features.csv - {len(df)} rows, {len(df.columns)} cols")
    
    # Process test.csv if exists
    test_file = src / "test.csv"
    if test_file.exists():
        df = pd.read_csv(test_file)
        df = create_features(df)
        df.to_csv(dst / "test_features.csv", index=False)
        print(f"Gold: test_features.csv - {len(df)} rows")

if __name__ == "__main__":
    main()

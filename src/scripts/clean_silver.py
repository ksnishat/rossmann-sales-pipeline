"""
Rossmann Sales DVC Pipeline - Silver Layer
Cleans and validates raw sales data.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/bronze", help="Input directory")
    parser.add_argument("--output", default="data/silver", help="Output directory")
    args = parser.parse_args()
    
    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    
    for csv_file in src.glob("*.csv"):
        df = pd.read_csv(csv_file)
        initial = len(df)
        
        # Basic cleaning
        df = df.drop_duplicates()
        after_dedup = len(df)
        
        # Handle missing values in sales
        if "Sales" in df.columns:
            df["Sales"] = df["Sales"].fillna(df["Sales"].median())
        
        # Remove outliers (sales > 100000 is likely data error)
        if "Sales" in df.columns:
            df = df[df["Sales"] <= 100000]
        
        after_outliers = len(df)
        out_file = dst / csv_file.name
        df.to_csv(out_file, index=False)
        
        print(f"Silver: {csv_file.name} - {initial} -> {after_outliers} rows "
              f"(removed {initial - after_dedup} dupes, {after_dedup - after_outliers} outliers)")

if __name__ == "__main__":
    main()

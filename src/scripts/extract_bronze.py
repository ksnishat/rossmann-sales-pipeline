"""
Rossmann Sales DVC Pipeline - Bronze Layer
Extracts raw sales data from CSV and databases.
"""
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw", help="Source CSV files")
    parser.add_argument("--output", default="data/bronze", help="Output directory")
    args = parser.parse_args()
    
    src = Path(args.source)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(src.glob("*.csv"))
    logger = __import__("logging").getLogger()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        logger.info(f"{csv_file.name}: {len(df)} rows, columns: {list(df.columns)[:5]}")
        # Save clean copy to bronze
        out_file = dst / csv_file.name
        df.to_csv(out_file, index=False)
        print(f"Bronze: {csv_file.name} -> {len(df)} rows")

if __name__ == "__main__":
    main()

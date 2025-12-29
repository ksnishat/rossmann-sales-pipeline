import pandas as pd
from sqlalchemy import create_engine

# 1. DATABASE CONNECTION
# We connect to the Docker container running on localhost port 5432
db_url = "postgresql://user:password@localhost:5432/rossmann"
engine = create_engine(db_url)

def load_file(file_path, table_name):
    print(f" Loading {file_path} into table '{table_name}'...")
    
    # 2. EXTRACT (Read from Disk)
    # low_memory=False: Tells Pandas to guess data types accurately, 
    # even if it uses more RAM. Essential for messy CSVs.
    df = pd.read_csv(file_path, low_memory=False)
    
    # 3. LOAD (Write to Database)
    # chunksize=10000: CRITICAL FOR INTERVIEWS.
    # Instead of pushing 1 million rows at once (which crashes servers),
    # we stream data in small batches of 10,000. It is safer and more stable.
    df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=10000)
    
    print(f" Success! Loaded {len(df)} rows into '{table_name}'.")

if __name__ == "__main__":
    # Load the Store info
    load_file("data/raw/store.csv", "store")
    
    # Load the Sales data (The big file)
    load_file("data/raw/train.csv", "train")

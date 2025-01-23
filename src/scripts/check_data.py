import pandas as pd
from sqlalchemy import create_engine, text

db_url = "postgresql://user:password@localhost:5432/rossmann"
engine = create_engine(db_url)

print("Checking Database Content...\n")

with engine.connect() as connection:
    # Check 1: Count rows (Did we lose any data?)
    result_train = connection.execute(text("SELECT COUNT(*) FROM train")).scalar()
    result_store = connection.execute(text("SELECT COUNT(*) FROM store")).scalar()

    print(f"✅ 'train' table has {result_train} rows (Expected: 1017209)")
    print(f"✅ 'store' table has {result_store} rows (Expected: 1115)")

    # Check 2: Preview data (Does it look like garbage?)
    print("\nPreview of 'store' table:")
    df = pd.read_sql("SELECT * FROM store LIMIT 3", connection)
    print(df.to_string())

print("\n Verification Complete!")

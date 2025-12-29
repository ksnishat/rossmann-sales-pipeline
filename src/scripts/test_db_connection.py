from sqlalchemy import create_engine, text

# 1. Define the connection string (The "Address" of the DB)
# Format: postgresql://user:password@localhost:port/database_name
db_url = "postgresql://user:password@localhost:5432/rossmann"

# 2. Create the connection engine
engine = create_engine(db_url)

# 3. Try to connect and run a simple query
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Success! Python is connected to the Docker Database.")
except Exception as e:
    print(f"❌ Error: {e}")

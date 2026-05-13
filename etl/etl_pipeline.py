import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Create database engine
engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as conn:
        print("Connected to Supabase PostgreSQL successfully!")
except Exception as e:
    print("Connection failed:")
    print(e)

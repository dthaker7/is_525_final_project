"""
db.py — Supabase / PostgreSQL connection helper
------------------------------------------------
Reads credentials from environment variables (or a .env file).
All other modules import `get_engine()` from here — never hardcode credentials.

Environment variables required (set in .env or GitHub Actions secrets):
    SUPABASE_HOST      e.g. db.xxxxxxxxxxxx.supabase.co
    SUPABASE_PORT      e.g. 5432
    SUPABASE_DB        e.g. postgres
    SUPABASE_USER      e.g. postgres
    SUPABASE_PASSWORD  your Supabase database password
"""

import os
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env file if it exists (ignored in production where env vars are injected)
load_dotenv()

logger = logging.getLogger(__name__)


def get_engine():
    """
    Returns a SQLAlchemy engine connected to Supabase PostgreSQL.
    Raises a clear error if any required env var is missing.
    """
    required = ["SUPABASE_HOST", "SUPABASE_PORT", "SUPABASE_DB",
                "SUPABASE_USER", "SUPABASE_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Create a .env file or set them in your environment / GitHub Actions secrets."
        )

    host     = os.environ["SUPABASE_HOST"]
    port     = os.environ["SUPABASE_PORT"]
    db       = os.environ["SUPABASE_DB"]
    user     = os.environ["SUPABASE_USER"]
    password = os.environ["SUPABASE_PASSWORD"]

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(url, pool_pre_ping=True)

    # Quick connectivity check
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("✓ Connected to Supabase PostgreSQL at %s:%s/%s", host, port, db)

    return engine

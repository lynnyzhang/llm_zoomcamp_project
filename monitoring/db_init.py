import os
from datetime import datetime

import psycopg

# Re-exported so `from monitoring.db_init import init_db, init_feedback` keeps
# working (deployment/entrypoint.sh imports them from here); the SQL and the
# functions themselves live in db_schema.
from .db_schema import init_db, init_feedback

DB_TIMEZONE = datetime.now().astimezone().tzinfo


def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "capstone"),
        user=os.getenv("POSTGRES_USER", "capstone"),
        password=os.getenv("POSTGRES_PASSWORD", "capstone_secret"),
    )

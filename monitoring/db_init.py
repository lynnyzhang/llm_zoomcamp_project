import logging
import os
from datetime import datetime

import psycopg

DB_TIMEZONE = datetime.now().astimezone().tzinfo

CONVERSATIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS conversations (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        course TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        response_time FLOAT NOT NULL,
        source TEXT,
        rejected INTEGER NOT NULL DEFAULT 0,
        span_id TEXT,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
    )
"""

SEARCHES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS searches (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id),
        span_id TEXT,
        query TEXT,
        search_query TEXT,
        source TEXT,
        results TEXT,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
    )
"""

LLM_CALLS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS llm_calls (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id),
        span_id TEXT,
        model TEXT,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        latency FLOAT,
        error TEXT,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
    )
"""

FEEDBACK_SCHEMA = """
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id),
        source TEXT NOT NULL,
        relevance TEXT,
        explanation TEXT,
        score INTEGER,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
    )
"""

MIGRATE_SESSION_ID = (
    "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS session_id TEXT"
)
MIGRATE_ERROR = "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS error TEXT"


def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "capstone"),
        user=os.getenv("POSTGRES_USER", "capstone"),
        password=os.getenv("POSTGRES_PASSWORD", "capstone_secret"),
    )


def init_db(drop=False):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS llm_calls")
                cur.execute("DROP TABLE IF EXISTS searches")
                cur.execute("DROP TABLE IF EXISTS conversations")
            cur.execute(CONVERSATIONS_SCHEMA)
            # Migrate databases created before these columns existed: sqlite has
            # no ADD COLUMN IF NOT EXISTS, but Postgres does (same pattern as the
            # span_id migration in tracer.py).
            cur.execute(MIGRATE_SESSION_ID)
            cur.execute(MIGRATE_ERROR)
            cur.execute(SEARCHES_SCHEMA)
            cur.execute(LLM_CALLS_SCHEMA)
        conn.commit()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to initialize conversations table", exc_info=True
        )
    finally:
        if conn is not None:
            conn.close()


def init_feedback(drop=False):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute(FEEDBACK_SCHEMA)
        conn.commit()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to initialize feedback table", exc_info=True
        )
    finally:
        if conn is not None:
            conn.close()

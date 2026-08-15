import pandas as pd

from monitoring.db_init import get_db_connection


def load_dataframe(query):
    # Opens a fresh connection per call: psycopg connections are thread-bound,
    # so a cached one would break reruns from a different thread.
    conn = get_db_connection()
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def postgres_reachable():
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except Exception:
        return False

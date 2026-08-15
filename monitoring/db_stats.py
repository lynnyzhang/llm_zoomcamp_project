import logging
from dataclasses import dataclass

from .db_init import get_db_connection


@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


def get_stats():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    AVG(response_time),
                    SUM(cost),
                    AVG(total_tokens)
                FROM conversations
            """)
            row = cur.fetchone()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load conversation stats", exc_info=True
        )
        return Stats(total=0, avg_response_time=0.0, total_cost=0.0, avg_tokens=0.0)
    finally:
        if conn is not None:
            conn.close()

    return Stats(
        total=row[0],
        avg_response_time=row[1] or 0.0,
        total_cost=row[2] or 0.0,
        avg_tokens=row[3] or 0.0,
    )


def get_user_feedback_stats():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END)
                FROM feedback
                WHERE source = 'user'
            """)
            row = cur.fetchone()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load user feedback stats", exc_info=True
        )
        return (0, 0)
    finally:
        if conn is not None:
            conn.close()

    return (row[0] or 0, row[1] or 0)

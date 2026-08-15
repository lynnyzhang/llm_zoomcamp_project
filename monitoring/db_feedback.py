import logging
from datetime import datetime

from .db_init import DB_TIMEZONE, get_db_connection


def save_feedback(conversation_id, source, relevance=None,
                  explanation=None, score=None):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (conversation_id, source, relevance,
                 explanation, score, timestamp),
            )
        conn.commit()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to save feedback", exc_info=True
        )
    finally:
        if conn is not None:
            conn.close()

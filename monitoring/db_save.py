import logging
from datetime import datetime

from .db_init import DB_TIMEZONE, get_db_connection


def save_conversation(record, question, course, session_id=None):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, course, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, source, rejected, span_id,
                    timestamp, session_id, error
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    course,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    record.source,
                    int(bool(record.rejected)),
                    record.span_id,
                    timestamp,
                    session_id,
                    record.error,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to save conversation", exc_info=True
        )
        return None
    finally:
        if conn is not None:
            conn.close()
    return conversation_id

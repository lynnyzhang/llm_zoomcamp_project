import logging

from .db_init import get_db_connection
from src.rag.metrics import LLMCallRecord


def row_to_record(row):
    return LLMCallRecord(
        id=row[0],
        question=row[1],
        answer=row[2],
        model=row[4],
        prompt_tokens=row[5],
        completion_tokens=row[6],
        total_tokens=row[7],
        response_time=row[8],
        cost=row[9],
        source=row[10],
        rejected=bool(row[11]),
        span_id=row[12],
        timestamp=row[13],
        error=row[15],
        prompt="",
        instructions="",
    )


def get_conversations(limit=10, session_id=None):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            sql = """
                SELECT id, question, answer, course, model,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, source, rejected, span_id,
                       timestamp, session_id, error
                FROM conversations
            """
            params = []
            if session_id is not None:
                sql += " WHERE session_id = %s"
                params.append(session_id)
            sql += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load conversations", exc_info=True
        )
        return []
    finally:
        if conn is not None:
            conn.close()

    return [row_to_record(row) for row in rows]


def get_feedback_for_conversations(conversation_ids):
    if not conversation_ids:
        return {}
    conn = None
    try:
        placeholders = ",".join("%s" for _ in conversation_ids)
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT conversation_id, score
                FROM feedback
                WHERE source = 'user'
                  AND conversation_id IN ({})
                """.format(placeholders),
                tuple(conversation_ids),
            )
            rows = cur.fetchall()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load conversation feedback", exc_info=True
        )
        return {}
    finally:
        if conn is not None:
            conn.close()

    return {row[0]: row[1] for row in rows if row[1] is not None}

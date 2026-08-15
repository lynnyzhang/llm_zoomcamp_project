import json
import logging
from datetime import datetime

from .db_init import DB_TIMEZONE, get_db_connection


def save_search(conversation_id, span_id, query, search_query, source, results):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO searches (
                    conversation_id, span_id, query, search_query, source,
                    results, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    conversation_id,
                    span_id,
                    query,
                    search_query,
                    source,
                    json.dumps(results, ensure_ascii=False),
                    timestamp,
                ),
            )
        conn.commit()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to save search", exc_info=True
        )
    finally:
        if conn is not None:
            conn.close()


def save_llm_call(conversation_id, span_id, model, prompt_tokens,
                  completion_tokens, total_tokens, latency, error):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO llm_calls (
                    conversation_id, span_id, model, prompt_tokens,
                    completion_tokens, total_tokens, latency, error, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    conversation_id,
                    span_id,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    latency,
                    error,
                    timestamp,
                ),
            )
        conn.commit()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to save LLM call", exc_info=True
        )
    finally:
        if conn is not None:
            conn.close()

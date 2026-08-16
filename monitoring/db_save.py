import json
import logging
from datetime import datetime

from src.rag.llm_call_record import LLMCallRecord

from .db_init import DB_TIMEZONE, get_db_connection


def save_conversation(
    record: LLMCallRecord, question: str, course: str, session_id: str | None = None
):
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


def save_search(
    conversation_id: int,
    span_id: str | None,
    query: str,
    search_query: str | None,
    source: str | None,
    results: list,
):
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
        logging.getLogger(__name__).warning("Failed to save search", exc_info=True)
    finally:
        if conn is not None:
            conn.close()


def save_llm_call(
    conversation_id: int,
    span_id: str | None,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    latency: float,
    error: str | None,
):
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
        logging.getLogger(__name__).warning("Failed to save LLM call", exc_info=True)
    finally:
        if conn is not None:
            conn.close()

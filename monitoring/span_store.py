import logging

from .db_init import get_db_connection
from .span_exporter import ensure_postgres_schema


def record_feedback(span_id, feedback):
    if not span_id:
        return False
    try:
        conn = get_db_connection()
        ensure_postgres_schema(conn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE spans SET feedback = %s WHERE span_id = %s",
                    (feedback, span_id),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO spans (span_id, feedback) VALUES (%s, %s)",
                        (span_id, feedback),
                    )
            conn.commit()
            recorded = True
        finally:
            conn.close()
    except Exception:
        logging.getLogger(__name__).warning("Failed to record feedback", exc_info=True)
        return False
    return recorded


def get_trace_stats():
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                stats = {}
                cur.execute("SELECT COUNT(*) FROM spans")
                stats["total_traces"] = cur.fetchone()[0]
                cur.execute("SELECT DISTINCT name FROM spans")
                stats["span_names"] = [r[0] for r in cur.fetchall()]
                cur.execute(
                    "SELECT SUM(input_tokens), SUM(output_tokens) FROM spans "
                    "WHERE input_tokens IS NOT NULL"
                )
                row = cur.fetchone()
                stats["total_input_tokens"] = row[0] or 0
                stats["total_output_tokens"] = row[1] or 0
                cur.execute(
                    "SELECT feedback, COUNT(*) FROM spans "
                    "WHERE feedback IS NOT NULL GROUP BY feedback"
                )
                stats["feedback"] = {r[0]: r[1] for r in cur.fetchall()}
        finally:
            conn.close()
    except Exception:
        logging.getLogger(__name__).warning("Failed to load trace stats", exc_info=True)
        return {"total_traces": 0}
    return stats

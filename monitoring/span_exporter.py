import logging

from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)

from .db_init import get_db_connection

PG_SPANS_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    name TEXT,
    start_time BIGINT,
    end_time BIGINT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    feedback TEXT DEFAULT NULL,
    agent_iterations INTEGER DEFAULT NULL,
    query TEXT DEFAULT NULL,
    search_queries TEXT DEFAULT NULL,
    span_id TEXT DEFAULT NULL
)
"""


def ensure_postgres_schema(conn):
    with conn.cursor() as cur:
        cur.execute(PG_SPANS_SCHEMA)
        cur.execute("ALTER TABLE spans ADD COLUMN IF NOT EXISTS span_id TEXT")
    conn.commit()


def span_id(span):
    # Stable hex span id shared by exporters and run_with_feedback.
    return format(span.get_span_context().span_id, "016x")


class PostgresSpanExporter(SpanExporter):
    """Export finished spans to a Postgres `spans` table."""

    # Every failure is contained — the exporter logs and returns FAILURE
    # instead of raising, so the app never crashes when Postgres is down.
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conn = get_db_connection()
        ensure_postgres_schema(self.conn)

    def export(self, spans):
        try:
            with self.conn.cursor() as cur:
                for span in spans:
                    attrs = dict(span.attributes or {})
                    cur.execute(
                        """INSERT INTO spans
                           (name, start_time, end_time, input_tokens,
                            output_tokens, feedback, agent_iterations,
                            query, search_queries, span_id)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            span.name,
                            span.start_time,
                            span.end_time,
                            attrs.get("input_tokens"),
                            attrs.get("output_tokens"),
                            attrs.get("feedback"),
                            attrs.get("agent_iterations"),
                            attrs.get("query"),
                            attrs.get("search_queries"),
                            span_id(span),
                        ),
                    )
            self.conn.commit()
        except Exception:
            self.logger.warning("Failed to export spans to Postgres", exc_info=True)
            self.conn.rollback()
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self):
        try:
            self.conn.close()
        except Exception:
            self.logger.warning("Postgres shutdown failed", exc_info=True)

    def force_flush(self):
        try:
            self.conn.commit()
        except Exception:
            self.logger.warning("Postgres force_flush failed", exc_info=True)
            return False
        return True

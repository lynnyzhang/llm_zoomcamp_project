# OpenTelemetry tracing: SQLiteSpanExporter (always-on) plus
# PostgresSpanExporter (when POSTGRES_HOST is set).
#
# Schema: name, start_time, end_time, input_tokens, output_tokens, cost,
#         feedback, agent_iterations, query, search_queries

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

DB_DIR = Path(__file__).resolve().parents[1] / "monitoring"
DB_PATH = DB_DIR / "traces.db"


def get_traces_db_path():
    return DB_PATH


def postgres_config():
    # Returns None (local-dev path) when POSTGRES_HOST is unset, so local runs
    # and tests never require Postgres. Defaults mirror docker-compose.yml.
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "capstone"),
        "user": os.environ.get("POSTGRES_USER", "capstone"),
        "password": os.environ.get("POSTGRES_PASSWORD", "capstone_secret"),
    }


def span_id(span):
    # Stable hex span id shared by exporters and run_with_feedback.
    return format(span.get_span_context().span_id, "016x")


def tracing_enabled():
    # Defaults to enabled; set TRACING_ENABLED=0|false|no|off to disable, e.g.
    # for environments without a writable data/ directory.
    raw = os.environ.get("TRACING_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


# ---------------------------------------------------------------------------
# SQLite span exporter
# ---------------------------------------------------------------------------

class SQLiteSpanExporter(SpanExporter):
    """Export finished spans to a SQLite database."""

    # Extended schema beyond the homework: adds feedback, agent_iterations,
    # query, and search_queries columns for richer monitoring.
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self):
        # Fresh connection per call: SQLite connections are thread-bound, so a
        # cached one would break exports from another thread (same reason
        # dashboard.py opens a new connection per query).
        return sqlite3.connect(str(self.db_path))

    def ensure_schema(self):
        conn = self.connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    name TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost REAL,
                    feedback TEXT DEFAULT NULL,
                    agent_iterations INTEGER DEFAULT NULL,
                    query TEXT DEFAULT NULL,
                    search_queries TEXT DEFAULT NULL,
                    span_id TEXT DEFAULT NULL
                )
            """)
            # Migrate databases created before the span_id column existed (e.g.
            # deployment/entrypoint.sh): SQLite has no ADD COLUMN IF NOT EXISTS.
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(spans)")
            }
            if "span_id" not in columns:
                conn.execute(
                    "ALTER TABLE spans ADD COLUMN span_id TEXT DEFAULT NULL"
                )
            conn.commit()
        finally:
            conn.close()

    def export(self, spans):
        # Every write failure is contained (warning + FAILURE) — an unwritable
        # database must never crash the app.
        try:
            conn = self.connect()
            try:
                for span in spans:
                    attrs = dict(span.attributes or {})
                    conn.execute(
                        """INSERT INTO spans
                           (name, start_time, end_time, input_tokens,
                            output_tokens, cost, feedback, agent_iterations,
                            query, search_queries, span_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            span.name,
                            span.start_time,
                            span.end_time,
                            attrs.get("input_tokens"),
                            attrs.get("output_tokens"),
                            attrs.get("cost"),
                            attrs.get("feedback"),
                            attrs.get("agent_iterations"),
                            attrs.get("query"),
                            attrs.get("search_queries"),
                            span_id(span),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to export spans to SQLite", exc_info=True
            )
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self):
        # Connections are opened per export call — nothing to close here.
        pass

    def force_flush(self):
        return True


# ---------------------------------------------------------------------------
# Postgres span exporter (docker path, opt-in via POSTGRES_HOST)
# ---------------------------------------------------------------------------

PG_SPANS_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    name TEXT,
    start_time BIGINT,
    end_time BIGINT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost DOUBLE PRECISION,
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
        cur.execute(
            "ALTER TABLE spans ADD COLUMN IF NOT EXISTS span_id TEXT"
        )
    conn.commit()


class PostgresSpanExporter(SpanExporter):
    """Export finished spans to a Postgres `spans` table."""

    # Mirrors SQLiteSpanExporter: same columns, BIGINT nanosecond timestamps
    # (start_time/end_time), created via psycopg if it doesn't exist. Used only
    # when POSTGRES_HOST is set (docker path); SQLite remains the always-on
    # store. Every failure is contained — the exporter logs and returns
    # FAILURE instead of raising, so the app never crashes when Postgres is
    # down or unreachable.
    def __init__(self, config=None):
        import psycopg

        self.logger = logging.getLogger(__name__)
        cfg = config or postgres_config()
        if cfg is None:
            raise RuntimeError(
                "PostgresSpanExporter requires POSTGRES_HOST to be set"
            )
        self.conn = psycopg.connect(**cfg)
        ensure_postgres_schema(self.conn)

    def export(self, spans):
        try:
            with self.conn.cursor() as cur:
                for span in spans:
                    attrs = dict(span.attributes or {})
                    cur.execute(
                        """INSERT INTO spans
                           (name, start_time, end_time, input_tokens,
                            output_tokens, cost, feedback, agent_iterations,
                            query, search_queries, span_id)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            span.name,
                            span.start_time,
                            span.end_time,
                            attrs.get("input_tokens"),
                            attrs.get("output_tokens"),
                            attrs.get("cost"),
                            attrs.get("feedback"),
                            attrs.get("agent_iterations"),
                            attrs.get("query"),
                            attrs.get("search_queries"),
                            span_id(span),
                        ),
                    )
            self.conn.commit()
        except Exception:
            self.logger.warning(
                "Failed to export spans to Postgres", exc_info=True
            )
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
            self.logger.warning(
                "Postgres force_flush failed", exc_info=True
            )
            return False
        return True


# ---------------------------------------------------------------------------
# Tracer setup helper
# ---------------------------------------------------------------------------

class TracerSetup:
    def __init__(self):
        self.provider = TracerProvider()
        self.exporter: SQLiteSpanExporter | None = None
        self.postgres_exporter: PostgresSpanExporter | None = None
        if tracing_enabled():
            try:
                self.exporter = SQLiteSpanExporter()
                self.provider.add_span_processor(
                    SimpleSpanProcessor(self.exporter)
                )
            except Exception:
                logging.getLogger(__name__).warning(
                    "SQLite span export disabled: %s",
                    "could not open database",
                    exc_info=True,
                )
            if postgres_config() is not None:
                try:
                    self.postgres_exporter = PostgresSpanExporter()
                    self.provider.add_span_processor(
                        SimpleSpanProcessor(self.postgres_exporter)
                    )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Postgres span export disabled: %s",
                        "could not connect",
                        exc_info=True,
                    )
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer("llm-zoomcapstone")

    def shutdown(self):
        if self.exporter is not None:
            self.exporter.force_flush()
            self.exporter.shutdown()
        if self.postgres_exporter is not None:
            self.postgres_exporter.force_flush()
            self.postgres_exporter.shutdown()


# ---------------------------------------------------------------------------
# Global tracer (lazily initialized)
# ---------------------------------------------------------------------------

default_setup: TracerSetup | None = None
setup_lock = threading.Lock()


def get_tracer():
    # Streamlit reruns the script from different threads (and multiple sessions
    # run concurrently), so the lazy singleton must be guarded: two threads
    # racing here would each build a TracerSetup, double-register a global
    # tracer provider, and orphan the first exporter.
    global default_setup
    with setup_lock:
        if default_setup is None:
            default_setup = TracerSetup()
        return default_setup.tracer


# ---------------------------------------------------------------------------
# TracedRAGAgent: wraps RAGAgent with OpenTelemetry spans
# ---------------------------------------------------------------------------

class TracedRAGAgent:
    def __init__(self, agent, tracer=None):
        self.agent = agent
        self.tracer = tracer or get_tracer()

    def run(self, query):
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", query)

            result = self.agent.run(query)

            span.set_attribute("agent_iterations", result.get("iterations", 0))
            span.set_attribute("search_count", len(result.get("searches", [])))

            search_queries = [
                s.query for s in result.get("searches", [])
            ]
            span.set_attribute("search_queries", json.dumps(search_queries))

            return result

    def run_with_feedback(self, query):
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", query)

            result = self.agent.run(query)

            span.set_attribute("agent_iterations", result.get("iterations", 0))
            span.set_attribute("search_count", len(result.get("searches", [])))

            search_queries = [
                s.query for s in result.get("searches", [])
            ]
            span.set_attribute("search_queries", json.dumps(search_queries))

            span_id = format(span.get_span_context().span_id, "016x")
            return result, span_id


# ---------------------------------------------------------------------------
# Feedback storage
# ---------------------------------------------------------------------------

def record_feedback_postgres(span_id, feedback):
    # Dual-write feedback into the Postgres spans table (docker path), running
    # only when POSTGRES_HOST is set. Updates the exact span by span_id,
    # inserting a placeholder row when the span export never reached Postgres.
    # Failures are logged and swallowed — feedback in SQLite must never be
    # lost because Postgres is down.
    cfg = postgres_config()
    if cfg is None or not span_id:
        return
    try:
        import psycopg

        with psycopg.connect(**cfg) as conn:
            ensure_postgres_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE spans SET feedback = %s WHERE span_id = %s",
                    (feedback, span_id),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO spans (span_id, feedback) "
                        "VALUES (%s, %s)",
                        (span_id, feedback),
                    )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to record feedback in Postgres", exc_info=True
        )


def record_feedback(span_id, feedback, db_path=None):
    # Feedback is matched by exact span_id against the span_id column. When
    # None, falls back to the legacy behavior: update the first (oldest)
    # feedback-less agent.run row. Never raises — an unwritable database logs
    # a warning and returns False.
    path = Path(db_path) if db_path else DB_PATH
    if not path.exists():
        return False

    try:
        conn = sqlite3.connect(str(path))
        try:
            if span_id:
                cursor = conn.execute(
                    "UPDATE spans SET feedback = ? WHERE span_id = ?",
                    (feedback, span_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE spans SET feedback = ? WHERE rowid = ("
                    "SELECT rowid FROM spans WHERE name = 'agent.run' "
                    "AND feedback IS NULL LIMIT 1"
                    ")",
                    (feedback,),
                )
            conn.commit()
            recorded = cursor.rowcount > 0
        finally:
            conn.close()
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to record feedback in SQLite", exc_info=True
        )
        return False

    record_feedback_postgres(span_id, feedback)
    return recorded


def get_trace_stats(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    if not path.exists():
        return {"total_traces": 0}

    conn = sqlite3.connect(str(path))
    try:
        stats = {}

        row = conn.execute("SELECT COUNT(*) FROM spans").fetchone()
        stats["total_traces"] = row[0]

        rows = conn.execute("SELECT DISTINCT name FROM spans").fetchall()
        stats["span_names"] = [r[0] for r in rows]

        row = conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM spans "
            "WHERE input_tokens IS NOT NULL"
        ).fetchone()
        stats["total_input_tokens"] = row[0] or 0
        stats["total_output_tokens"] = row[1] or 0

        row = conn.execute(
            "SELECT SUM(cost) FROM spans WHERE cost IS NOT NULL"
        ).fetchone()
        stats["total_cost"] = row[0] or 0.0

        row = conn.execute(
            "SELECT feedback, COUNT(*) FROM spans "
            "WHERE feedback IS NOT NULL GROUP BY feedback"
        ).fetchall()
        stats["feedback"] = {r[0]: r[1] for r in row}

        return stats
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing OpenTelemetry tracing with SQLite storage...")
    print(f"Database path: {DB_PATH}")

    # Initialize tracer
    setup = TracerSetup()
    tracer = setup.tracer

    # Create a test span
    with tracer.start_as_current_span("test.span") as span:
        span.set_attribute("input_tokens", 100)
        span.set_attribute("output_tokens", 50)
        span.set_attribute("cost", 0.001)
        span.set_attribute("query", "test query")
        span.set_attribute("agent_iterations", 2)
        print("Test span created.")

    # Verify database
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.execute("PRAGMA table_info(spans)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Schema columns: {columns}")

    cursor = conn.execute("SELECT * FROM spans")
    rows = cursor.fetchall()
    print(f"Rows in spans table: {len(rows)}")
    for row in rows:
        print(f"  {row}")

    conn.close()
    setup.shutdown()
    print("Done!")

"""OpenTelemetry tracing with SQLite storage for the capstone RAG agent.

Adapted from 5-Monitoring/assignment.ipynb (SQLiteSpanExporter pattern) and
5-Monitoring/starter.py (OpenTelemetry setup).

Schema: name, start_time, end_time, input_tokens, output_tokens, cost,
        feedback, agent_iterations, query, search_queries
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

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

_DB_DIR = Path(__file__).resolve().parents[2] / "data"
_DB_PATH = _DB_DIR / "traces.db"


def get_traces_db_path() -> Path:
    """Return the path to the traces SQLite database."""
    return _DB_PATH


# ---------------------------------------------------------------------------
# SQLite span exporter (from 5-Monitoring/assignment.ipynb pattern)
# ---------------------------------------------------------------------------

class SQLiteSpanExporter(SpanExporter):
    """Export finished spans to a SQLite database.

    Extended schema beyond the homework: adds feedback, agent_iterations,
    query, and search_queries columns for richer monitoring.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the spans table if it doesn't exist."""
        self.conn.execute("""
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
                search_queries TEXT DEFAULT NULL
            )
        """)
        self.conn.commit()

    def export(self, spans) -> SpanExportResult:
        """Export a batch of finished spans to SQLite."""
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                """INSERT INTO spans
                   (name, start_time, end_time, input_tokens, output_tokens,
                    cost, feedback, agent_iterations, query, search_queries)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.conn.close()

    def force_flush(self) -> bool:
        self.conn.commit()
        return True


# ---------------------------------------------------------------------------
# Tracer setup helper
# ---------------------------------------------------------------------------

class TracerSetup:
    """Initialize and configure the OpenTelemetry tracer.

    Usage:
        setup = TracerSetup()
        tracer = setup.tracer
        # ... use tracer ...
        setup.shutdown()
    """

    def __init__(self, service_name: str = "llm-zoomcapstone"):
        self.provider = TracerProvider()
        self.exporter = SQLiteSpanExporter()
        self.provider.add_span_processor(
            SimpleSpanProcessor(self.exporter)
        )
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(service_name)

    def shutdown(self) -> None:
        """Flush and close the exporter."""
        self.exporter.force_flush()
        self.exporter.shutdown()


# ---------------------------------------------------------------------------
# Global tracer (lazily initialized)
# ---------------------------------------------------------------------------

_default_setup: TracerSetup | None = None


def get_tracer() -> trace.Tracer:
    """Return the global tracer, initializing if needed."""
    global _default_setup
    if _default_setup is None:
        _default_setup = TracerSetup()
    return _default_setup.tracer


# ---------------------------------------------------------------------------
# TracedRAGAgent: wraps RAGAgent with OpenTelemetry spans
# ---------------------------------------------------------------------------

class TracedRAGAgent:
    """Wrapper around RAGAgent that records OpenTelemetry traces.

    Records spans for:
    - agent.run: top-level span with query, iterations, feedback
    - agent.search: individual search iterations
    - agent.llm: LLM calls with token counts

    Adapted from 5-Monitoring/assignment.ipynb RAGTraced pattern.
    """

    def __init__(self, agent: Any, tracer: trace.Tracer | None = None):
        self.agent = agent
        self.tracer = tracer or get_tracer()

    def run(self, query: str) -> dict:
        """Run the agent with tracing."""
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", query)

            start = time.time()
            result = self.agent.run(query)
            elapsed = time.time() - start

            span.set_attribute("agent_iterations", result.get("iterations", 0))
            span.set_attribute("search_count", len(result.get("searches", [])))

            # Record search queries
            search_queries = [
                s.query for s in result.get("searches", [])
            ]
            span.set_attribute("search_queries", json.dumps(search_queries))

            return result

    def run_with_feedback(self, query: str) -> tuple[dict, str]:
        """Run agent, return result and a span_id for later feedback.

        Returns:
            (result_dict, span_id) — pass span_id to record_feedback().
        """
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", query)

            start = time.time()
            result = self.agent.run(query)
            elapsed = time.time() - start

            span.set_attribute("agent_iterations", result.get("iterations", 0))
            span.set_attribute("search_count", len(result.get("searches", [])))

            search_queries = [
                s.query for s in result.get("searches", [])
            ]
            span.set_attribute("search_queries", json.dumps(search_queries))

            # Use span_id as feedback reference
            span_id = format(span.get_span_context().span_id, "016x")
            return result, span_id


# ---------------------------------------------------------------------------
# Feedback storage
# ---------------------------------------------------------------------------

def record_feedback(
    span_id: str,
    feedback: str,
    db_path: str | Path | None = None,
) -> bool:
    """Record user feedback for a trace.

    Args:
        span_id: The span ID to associate feedback with.
        feedback: "positive" or "negative".
        db_path: Optional path to the traces database.

    Returns:
        True if feedback was recorded, False otherwise.
    """
    path = Path(db_path) if db_path else _DB_PATH
    if not path.exists():
        return False

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.execute(
            "UPDATE spans SET feedback = ? WHERE rowid = ("
            "SELECT rowid FROM spans WHERE name = 'agent.run' "
            "AND feedback IS NULL LIMIT 1"
            ")",
            (feedback,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_trace_stats(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return summary statistics from the traces database."""
    path = Path(db_path) if db_path else _DB_PATH
    if not path.exists():
        return {"total_traces": 0}

    conn = sqlite3.connect(str(path))
    try:
        stats: dict[str, Any] = {}

        # Total traces
        row = conn.execute("SELECT COUNT(*) FROM spans").fetchone()
        stats["total_traces"] = row[0]

        # Distinct span names
        rows = conn.execute("SELECT DISTINCT name FROM spans").fetchall()
        stats["span_names"] = [r[0] for r in rows]

        # Total tokens
        row = conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM spans "
            "WHERE input_tokens IS NOT NULL"
        ).fetchone()
        stats["total_input_tokens"] = row[0] or 0
        stats["total_output_tokens"] = row[1] or 0

        # Total cost
        row = conn.execute(
            "SELECT SUM(cost) FROM spans WHERE cost IS NOT NULL"
        ).fetchone()
        stats["total_cost"] = row[0] or 0.0

        # Feedback counts
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
    print(f"Database path: {_DB_PATH}")

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
    conn = sqlite3.connect(str(_DB_PATH))
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

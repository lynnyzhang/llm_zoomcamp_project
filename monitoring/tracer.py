# OpenTelemetry tracing setup: configures a global tracer backed by the
# Postgres span store (the only runtime store for all production data).

import logging
import os
import threading

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .exporter import PostgresSpanExporter
# Re-import so `from monitoring.tracer import TracedRAGAgent` keeps working
# (app.py imports it from here); the class itself lives in traced_agent.
from .traced_agent import TracedRAGAgent


def tracing_enabled():
    # Defaults to enabled; set TRACING_ENABLED=0|false|no|off to disable, e.g.
    # for environments without a writable store.
    raw = os.environ.get("TRACING_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


class TracerSetup:
    def __init__(self):
        self.provider = TracerProvider()
        self.exporter: PostgresSpanExporter | None = None
        if tracing_enabled():
            try:
                self.exporter = PostgresSpanExporter()
                self.provider.add_span_processor(
                    SimpleSpanProcessor(self.exporter)
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

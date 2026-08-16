# OpenTelemetry tracing setup: configures a global tracer backed by the
# Postgres span store (the only runtime store for all production data).

import json
import logging
import os
import threading

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from src.rag.llm_call_record import calculate_cost
from src.rag.rag_agent import RAGAgent
from src.rag.scoring import AgentResult

from .span_exporter import PostgresSpanExporter, span_id


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
                self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
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


class TracedRAGAgent:
    def __init__(self, agent: RAGAgent, tracer=None):
        if tracer is None:
            tracer = get_tracer()
        self.agent = agent
        self.tracer = tracer

    @property
    def agent_loop_record(self):
        # The saver reads the run's records from the agent it is given; the
        # wrapper delegates to the inner agent that actually ran the loop.
        return self.agent.agent_loop_record

    @property
    def calls(self):
        return self.agent.calls

    def run(self, query: str) -> AgentResult:
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", query)

            result = self.agent.run(query)

            span.set_attribute("agent_iterations", result.iterations)
            span.set_attribute("search_count", len(result.searches))

            search_queries = [s.query for s in result.searches]
            span.set_attribute("search_queries", json.dumps(search_queries))

            usage = result.usage
            if usage:
                span.set_attribute("input_tokens", usage.input_tokens)
                span.set_attribute("output_tokens", usage.output_tokens)
                span.set_attribute(
                    "cost",
                    calculate_cost(getattr(self.agent, "model", "") or "", usage),
                )

            self.agent.attach_span(span_id(span))
            return result

    def run_with_feedback(self, query: str) -> tuple[AgentResult, str]:
        with self.tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("query", query)

            result = self.agent.run(query)

            span.set_attribute("agent_iterations", result.iterations)
            span.set_attribute("search_count", len(result.searches))

            search_queries = [s.query for s in result.searches]
            span.set_attribute("search_queries", json.dumps(search_queries))

            usage = result.usage
            if usage:
                span.set_attribute("input_tokens", usage.input_tokens)
                span.set_attribute("output_tokens", usage.output_tokens)
                span.set_attribute(
                    "cost",
                    calculate_cost(getattr(self.agent, "model", "") or "", usage),
                )
            sid = span_id(span)
            self.agent.attach_span(sid)
            return result, sid

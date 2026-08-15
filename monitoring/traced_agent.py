import json

from src.rag.metrics import calculate_cost

from .exporter import span_id


class TracedRAGAgent:
    def __init__(self, agent, tracer=None):
        if tracer is None:
            # Lazy import: get_tracer lives in tracer.py which re-exports this
            # class, so a top-level import would cycle back to traced_agent.
            from .tracer import get_tracer
            tracer = get_tracer()
        self.agent = agent
        self.tracer = tracer

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

            usage = result.get("usage") or {}
            if usage:
                span.set_attribute("input_tokens", usage.get("input_tokens", 0))
                span.set_attribute("output_tokens", usage.get("output_tokens", 0))
                span.set_attribute("cost", calculate_cost(getattr(self.agent, "model", "") or "", usage))

            self.agent.attach_span(span_id(span))
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

            usage = result.get("usage") or {}
            if usage:
                span.set_attribute("input_tokens", usage.get("input_tokens", 0))
                span.set_attribute("output_tokens", usage.get("output_tokens", 0))
                span.set_attribute("cost", calculate_cost(getattr(self.agent, "model", "") or "", usage))
            sid = span_id(span)
            self.agent.attach_span(sid)
            return result, sid

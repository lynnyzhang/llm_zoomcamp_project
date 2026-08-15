import logging
import re
import time

from src.llm import LLMClient
from src.rag.RAGBase import RAGBase
from src.rag.execution import apply_tool_calls
from src.rag.metrics import LLMCallRecord
from src.rag.prompts import INSTRUCTIONS, REJECTION_MESSAGE
from src.rag.scoring import finalize_result, get_confidence_threshold
from src.rag.tools import LOCAL_SEARCH_TOOL, TOOLS
from src.search.hybrid import HybridSearch

MAX_ITERATIONS = 3

class RAGAgent(RAGBase):

    def __init__(self, search_index=None, llm_client=None, model=None,
                 max_iterations=MAX_ITERATIONS, num_results=5,
                 confidence_threshold=None):
        if search_index is None:
            search_index = HybridSearch()
        model = model or LLMClient.get_model()
        super().__init__(search_index=search_index, llm_client=llm_client, model=model)
        self.max_iterations = max_iterations
        self.num_results = num_results
        # Reuses the index embedder for grounding/relevance scores.
        self.embedder = getattr(search_index, "embedder", None)
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_confidence_threshold()
        )
        # Per-run recording: per-call records plus the turn record.
        self.calls: list[LLMCallRecord] = []
        self.turn_record: LLMCallRecord | None = None

    def perform_search(self, query, num_results=5):
        return self.search(query, num_results=num_results)

    def call_llm(self, items, tools, temperature):
        # Per-call recording: timing + usage land in self.calls as typed
        # records, so monitoring needs no raw dicts from the loop.
        start = time.perf_counter()
        try:
            response = super().call_llm(items, tools=tools, temperature=temperature)
            self.calls.append(LLMCallRecord.from_response(
                self.model, response, time.perf_counter() - start))
            return response
        except Exception:
            self.calls.append(LLMCallRecord.call_failed(
                self.model, time.perf_counter() - start))
            raise

    def record_turn(self, result, start):
        usage = {
            "input_tokens": sum(c.prompt_tokens or 0 for c in self.calls),
            "output_tokens": sum(c.completion_tokens or 0 for c in self.calls),
        }
        result["usage"] = usage
        result["llm_calls"] = [
            {"model": c.model, "prompt_tokens": c.prompt_tokens,
             "completion_tokens": c.completion_tokens, "total_tokens": c.total_tokens,
             "latency": c.response_time, "error": c.error} for c in self.calls
        ]
        self.turn_record = LLMCallRecord.turn(
            self.model, result, usage, time.perf_counter() - start)
        return result

    def attach_span(self, span_id):
        # The tracing layer tells the agent which span covered this turn; the
        # agent updates its own record instead of the caller mutating it.
        if self.turn_record is not None:
            self.turn_record.span_id = span_id

    def run(self, query):
        self.calls = []
        self.turn_record = None
        start = time.perf_counter()
        # Deterministic guard: empty/whitespace or pure punctuation can never
        # be a question — reject without spending an LLM round trip.
        if not query or not query.strip() or not re.search(r"[a-zA-Z0-9]", query):
            return self.record_turn({
                "answer": REJECTION_MESSAGE, "searches": [], "iterations": 0,
                "rejected": True, "source": None, "confidence": None,
                "relevance": None,
            }, start)

        items = [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": query},
        ]
        searches = []
        sources = set()
        try:
            for turn in range(self.max_iterations + 1):
                # The local knowledge base is the only tool on the first
                # turn, so the model cannot skip it; the web tool unlocks
                # afterwards while the model keeps all decisions.
                tools = TOOLS if turn > 0 else [LOCAL_SEARCH_TOOL]
                response = self.call_llm(items, tools, LLMClient.get_agent_temperature())
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    return self.record_turn(finalize_result(
                        response.output_text, query, searches, sources,
                        self.embedder, self.confidence_threshold), start)
                apply_tool_calls(self, calls, query, items, searches, sources)
        except Exception:
            # Fail-soft: an LLM/network failure must never fabricate an answer.
            logging.getLogger(__name__).warning(
                "Agent loop failed for %r", query, exc_info=True
            )
        # Loop exhausted without a final answer: never guess.
        return self.record_turn(finalize_result(
            None, query, searches, sources, self.embedder, self.confidence_threshold),
            start)

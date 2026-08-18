import logging
import re
import time

from src.llm_client import LLMClient
from src.rag.rag_base import RAGBase
from src.rag.llm_call_record import LLMCallSummary, Usage
from src.rag.llm_call_record import LLMCallRecord
from src.rag.prompts import ESCALATION_MESSAGE, INSTRUCTIONS, REJECTION_MESSAGE
from src.rag.scoring import AgentResult, finalize_result, get_confidence_threshold
from src.rag.tools import apply_tool_calls
from src.rag.tools import LOCAL_SEARCH_TOOL, TOOLS

MAX_ITERATIONS = 5


class RAGAgent(RAGBase):
    def __init__(
        self,
        search_index,
        llm_client=None,
        model=None,
        max_iterations=MAX_ITERATIONS,
        num_results=5,
        confidence_threshold=None,
    ):
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
        # Per-agent-loop recording: per-call records plus the loop record.
        self.calls: list[LLMCallRecord] = []
        self.agent_loop_record: LLMCallRecord | None = None

    def call_llm(
        self,
        messages: list[dict],
        tools=None,
        temperature=None,
    ):
        # Per-call recording: timing + usage land in self.calls as typed
        # records, so monitoring needs no raw dicts from the loop.
        temperature = (
            temperature
            if temperature is not None
            else LLMClient.get_agent_temperature()
        )
        start = time.perf_counter()
        try:
            response = super().call_llm(messages, tools=tools, temperature=temperature)
            self.calls.append(
                LLMCallRecord.from_response(
                    self.model, response, time.perf_counter() - start
                )
            )
            return response
        except Exception:
            self.calls.append(
                LLMCallRecord.call_failed(self.model, time.perf_counter() - start)
            )
            raise

    def record_agent_loop(self, result: AgentResult, start: float) -> AgentResult:
        usage = Usage(
            input_tokens=sum(c.prompt_tokens or 0 for c in self.calls),
            output_tokens=sum(c.completion_tokens or 0 for c in self.calls),
        )
        result.usage = usage
        result.llm_calls = [LLMCallSummary.from_record(c) for c in self.calls]
        self.agent_loop_record = LLMCallRecord.agent_loop(
            self.model, result, usage, time.perf_counter() - start
        )
        return result

    def attach_span(self, span_id: str):
        # The tracing layer tells the agent which span covered this agent
        # loop; the agent updates its own record instead of the caller
        # mutating it.
        if self.agent_loop_record is not None:
            self.agent_loop_record.span_id = span_id

    def run(self, query: str) -> AgentResult:
        return self.run_agent_loop(query)

    def run_agent_loop(self, query: str) -> AgentResult:
        self.calls = []
        self.agent_loop_record = None
        # Every grounding-gate outcome in order: the rejected attempt that
        # triggered escalation is retained here for monitoring/troubleshooting.
        self.gate_history: list[AgentResult] = []
        start = time.perf_counter()
        # Deterministic guard: empty/whitespace or pure punctuation can never
        # be a question — reject without spending an LLM round trip.
        if not query or not query.strip() or not re.search(r"[a-zA-Z0-9]", query):
            return self.record_agent_loop(AgentResult.rejected_result([]), start)

        messages = [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": query},
        ]
        searches = []
        sources = set()
        llm_api_call = 0
        escalated = False
        try:
            while llm_api_call < self.max_iterations + 1 + (1 if escalated else 0):
                # The local knowledge base is the only tool on the first LLM
                # API call, so the model cannot skip it; the web tool unlocks
                # afterwards while the model keeps all decisions.
                tools = TOOLS if llm_api_call > 0 else [LOCAL_SEARCH_TOOL]
                response = self.call_llm(
                    messages, tools, LLMClient.get_agent_temperature()
                )
                calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                if not calls:
                    answer = (response.output_text or "").strip()
                    result = finalize_result(
                        answer,
                        query,
                        searches,
                        sources,
                        self.embedder,
                        self.confidence_threshold,
                    )
                    self.gate_history.append(result)
                    if (
                        result.rejected
                        and answer
                        and answer != REJECTION_MESSAGE
                        and "web" not in sources
                        and not escalated
                    ):
                        # Fail-safe edge case: the model answered with a
                        # partial or irrelevant answer instead of calling the
                        # web tool itself, so the grounding gate rejected it.
                        # Force one Bulbapedia retry so a web-grounded answer
                        # is not starved. User role: the Responses API only
                        # allows developer messages at the start, and local
                        # chat templates reject a mid-conversation system
                        # message.
                        messages.append({"role": "user", "content": ESCALATION_MESSAGE})
                        escalated = True
                    else:
                        result.escalated = escalated
                        return self.record_agent_loop(result, start)
                else:
                    apply_tool_calls(self, calls, query, messages, searches, sources)
                llm_api_call += 1
        except Exception:
            # Fail-soft: an LLM/network failure must never fabricate an answer.
            logging.getLogger(__name__).warning(
                "Agent loop failed for %r", query, exc_info=True
            )
        # Loop exhausted without a final answer: never guess.
        final = finalize_result(
            None, query, searches, sources, self.embedder, self.confidence_threshold
        )
        final.escalated = escalated
        return self.record_agent_loop(final, start)

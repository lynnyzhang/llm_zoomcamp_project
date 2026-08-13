import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from src.llm import LLMClient
from src.rag.RAGBase import RAGBase
from src.search import web
from src.search.hybrid import HybridSearch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REJECTION_MESSAGE = (
    "I'm a Pokémon knowledge assistant — I can answer questions about Pokémon "
    "stats, types, weaknesses, abilities, evolutions, and type matchups. I "
    "can't predict battle outcomes, access save files, help with cheating, or "
    "answer non-Pokémon topics. Try asking about a specific Pokémon!"
)

INSTRUCTIONS = f"""\
You are a Pokémon knowledge assistant that answers questions by judging
retrieved context. For each question, decide ONE of four verdicts:

- "answer": the retrieved context fully answers the question. Write
  "answer" using ONLY the context — never from memory.
- "answer_partial": the retrieved context answers the question only
  partially. Write a grounded partial answer: state what the context
  supports, then state what it does not determine. Example: for "which
  Pokémon pair well with Pikachu?", if the context shows Pikachu is
  Electric and weak to Ground, answer "Pikachu is Electric-type; based on
  type coverage, partners that resist Ground complement it — the context
  does not rank specific best partners." Questions asking for "best"/"top"
  picks work the same way: give the grounded guidance and explicitly say
  the context does not establish a definitive ranking. A grounded partial
  answer always beats "escalate". The system will try a web search to
  complete the answer. Hedge unsupported claims ("based on the retrieved
  data", "the context does not say") and never invent facts.
- "escalate": the context contains no relevant facts for the question —
  empty, unrelated, or nothing usable. Only escalate when even a hedged
  partial answer would be ungrounded.
- "reject": the question asks the assistant to predict winners or simulate
  the outcome of a specific battle or tournament (e.g. "who would win
  Charizard vs Blastoise?"), access save files, provide cheats, emulators,
  or real-time data, or is not about Pokémon at all — coursework, finance,
  medicine, or any other non-Pokémon topic.
  Pokémon questions about type matchups, weaknesses, or team composition
  are NEVER rejected — even when phrased with battle formats ("3v3 battle",
  "gen 1"): use "answer" or "answer_partial" when the context supports it,
  otherwise "escalate".
  Pokémon-adjacent topics (game history, anime, manga, lore, characters)
  are also in scope: if the context lacks the answer, use "escalate",
  never "reject".

confidence: how confident you are that every claim in the answer is
grounded in the retrieved context (0.0 to 1.0) — groundedness, not
completeness: a partial answer stating exactly what the context does and
does not say can be highly confident. If verdict is "answer" or
"answer_partial", confidence must be at least the configured threshold.

The assistant answers questions about the 1,350-Pokémon knowledge base: stats,
types, weaknesses, abilities, evolutions, alternate forms, and type-matchup
guidance grounded in type coverage. It may escalate to Bulbapedia via web
search when the local knowledge base is insufficient. Answers come only from
retrieved context, never from memory. It cannot predict battle outcomes,
simulate battles, access save files, help with cheats or emulators, answer
non-Pokémon topics, or provide real-time data. When neither the local
knowledge base nor Bulbapedia yields a confident answer, reply with the
rejection message verbatim — never guess:
{REJECTION_MESSAGE}
"""

# Back-compat alias used by older callers/tests.
AGENT_INSTRUCTIONS = INSTRUCTIONS

MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Judge schema
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    verdict: Literal["answer", "answer_partial", "escalate", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)
    answer: str | None = None

    @model_validator(mode="after")
    def require_answer_for_answer(self):
        if self.verdict in ("answer", "answer_partial") and not self.answer:
            raise ValueError("answer is required when verdict is 'answer' or 'answer_partial'")
        return self


# ---------------------------------------------------------------------------
# Agent state / records
# ---------------------------------------------------------------------------


@dataclass
class SearchRecord:
    query: str
    results: list[dict]
    analysis: dict[str, Any] | None = None
    source: str | None = None


@dataclass
class AgentResult:
    answer: str
    searches: list[SearchRecord]
    iterations: int


class AgentState(TypedDict):
    query: str
    local_record: SearchRecord | None
    web_record: SearchRecord | None
    source: str | None
    confidence: float | None
    judge_answer: str | None
    partial_answer: str | None
    partial_confidence: float | None
    answer: str | None
    rejected: bool
    searches: list[SearchRecord]
    iterations: int


def get_confidence_threshold():
    return float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))


# ---------------------------------------------------------------------------
# RAGAgent (LangGraph escalate flow)
# ---------------------------------------------------------------------------


class RAGAgent:

    def __init__(
        self,
        search_index=None,
        llm_client=None,
        model=None,
        max_iterations=MAX_ITERATIONS,
        search_type="hybrid",
        num_results=5,
        confidence_threshold=None,
    ):
        if search_index is None:
            search_index = HybridSearch()
        model = model or LLMClient.get_model()
        self.rag = RAGBase(
            search_index=search_index,
            llm_client=llm_client,
            model=model,
            search_type=search_type,
        )
        self.llm_client = self.rag.llm_client
        self.model = model
        self.max_iterations = max_iterations
        self.search_type = search_type
        self.num_results = num_results
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_confidence_threshold()
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def perform_search(self, query, num_results=5):
        return self.rag.search(query, num_results=num_results)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("local_search", self.local_search)
        graph.add_node("local_judge", self.local_judge)
        graph.add_node("web_search", self.web_search_node)
        graph.add_node("web_judge", self.web_judge)
        graph.add_node("answer_node", self.answer_node)
        graph.add_node("reject_node", self.reject_node)
        graph.add_node("fallback_node", self.fallback_node)
        graph.add_node("finalize", self.finalize_node)

        graph.add_edge(START, "local_search")
        graph.add_conditional_edges(
            "local_search",
            self._route_local_start,
            {"judge": "local_judge", "empty": "web_search"},
        )
        graph.add_conditional_edges(
            "local_judge",
            self._route_local_judge,
            {
                "answer": "answer_node",
                "partial": "web_search",
                "escalate": "web_search",
                "reject": "reject_node",
            },
        )
        graph.add_edge("web_search", "web_judge")
        graph.add_conditional_edges(
            "web_judge",
            self._route_web_judge,
            {"answer": "answer_node", "fallback": "fallback_node"},
        )
        graph.add_edge("answer_node", "finalize")
        graph.add_edge("reject_node", "finalize")
        graph.add_edge("fallback_node", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_local_start(self, state):
        # Empty local results cannot be judged — go straight to the web.
        record = state.get("local_record")
        return "judge" if record and record.results else "empty"

    def _route_local_judge(self, state):
        analysis = state["local_record"].analysis or {}
        if (
            analysis.get("verdict") == "answer"
            and (analysis.get("confidence") or 0) >= self.confidence_threshold
        ):
            return "answer"
        if (
            analysis.get("verdict") == "answer_partial"
            and (analysis.get("confidence") or 0) >= self.confidence_threshold
        ):
            return "partial"
        if analysis.get("verdict") == "reject":
            return "reject"
        return "escalate"

    def _route_web_judge(self, state):
        analysis = state["web_record"].analysis or {}
        # The web path is the last stop: a grounded answer_partial from the
        # web judge is still the best available answer.
        if (
            analysis.get("verdict") in ("answer", "answer_partial")
            and (analysis.get("confidence") or 0) >= self.confidence_threshold
        ):
            return "answer"
        return "fallback"

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def local_search(self, state):
        results = self.perform_search(state["query"], num_results=self.num_results)
        record = SearchRecord(
            query=state["query"],
            results=results,
            analysis={"sufficient": bool(results), "verdict": None, "confidence": None},
            source="local",
        )
        return {"local_record": record}

    def local_judge(self, state):
        record = state["local_record"]
        try:
            verdict = self._judge(state["query"], record.results, "local knowledge base")
            if verdict is None:
                raise ValueError("judge returned no structured verdict")
            record.analysis = {
                "sufficient": True,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
            }
            updates = {
                "local_record": record,
                "source": "local",
                "confidence": verdict.confidence,
                "judge_answer": verdict.answer,
            }
            # Carry the grounded partial answer so the web path can fall back
            # to it when Bulbapedia cannot complete it.
            if (
                verdict.verdict == "answer_partial"
                and verdict.confidence >= self.confidence_threshold
            ):
                updates["partial_answer"] = verdict.answer
                updates["partial_confidence"] = verdict.confidence
            return updates
        except Exception:
            # Fail toward escalation: a judge failure must never fabricate an
            # answer or block the web fallback.
            record.analysis = {"sufficient": True, "verdict": "escalate", "confidence": 0.0}
            return {
                "local_record": record,
                "source": "local",
                "confidence": 0.0,
                "judge_answer": None,
            }

    def web_search_node(self, state):
        try:
            results = web.web_search(state["query"], num_results=self.num_results)
        except Exception:
            # Broad except: any web failure (missing/invalid key, usage limit,
            # network) yields empty results so the flow rejects gracefully.
            logging.getLogger(__name__).warning(
                "Web search failed for %r", state["query"], exc_info=True
            )
            results = []
        record = SearchRecord(
            query=state["query"],
            results=results,
            analysis={"sufficient": bool(results), "verdict": None, "confidence": None},
            source="web",
        )
        return {"web_record": record}

    def web_judge(self, state):
        record = state["web_record"]
        if not record.results:
            record.analysis = {
                "sufficient": False,
                "verdict": "reject",
                "confidence": 0.0,
            }
            return {
                "web_record": record,
                "source": "web",
                "confidence": 0.0,
                "judge_answer": None,
            }
        try:
            verdict = self._judge(state["query"], record.results, "Bulbapedia web results")
            if verdict is None:
                raise ValueError("judge returned no structured verdict")
            record.analysis = {
                "sufficient": True,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
            }
            return {
                "web_record": record,
                "source": "web",
                "confidence": verdict.confidence,
                "judge_answer": verdict.answer,
            }
        except Exception:
            # Final path: a judge failure here can only degrade to a rejection.
            record.analysis = {"sufficient": True, "verdict": "reject", "confidence": 0.0}
            return {
                "web_record": record,
                "source": "web",
                "confidence": 0.0,
                "judge_answer": None,
            }

    def answer_node(self, state):
        return {"answer": state.get("judge_answer") or "", "rejected": False}

    def reject_node(self, state):
        return {
            "answer": REJECTION_MESSAGE,
            "source": None,
            "confidence": None,
            "rejected": True,
        }

    def fallback_node(self, state):
        # Web could not answer confidently: surface the carried partial
        # answer instead of a rejection; without one, plain reject.
        if state.get("partial_answer"):
            return {
                "answer": state["partial_answer"],
                "source": "local",
                "confidence": state.get("partial_confidence"),
                "rejected": False,
            }
        return self.reject_node(state)

    def finalize_node(self, state):
        searches = [
            r
            for r in (state.get("local_record"), state.get("web_record"))
            if r is not None
        ]
        return {"searches": searches, "iterations": len(searches)}

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    def _judge(self, query, results, source_label):
        context = self._format_context(results, source_label)
        messages = [
            {"role": "developer", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\nRetrieved context:\n{context}\n\n"
                    'Reply with ONLY a JSON object: {"verdict": "answer"|"escalate"|"reject", '
                    '"confidence": <0.0-1.0>, "answer": <the answer only when verdict is "answer", else null>}'
                ),
            },
        ]
        # Skip the structured-output attempt when the one-time client test
        # already proved this server ignores it (llama.cpp ~24s wasted per
        # judge call) — the fallback below parses the raw text instead.
        temperature = LLMClient.get_judge_temperature()
        client = self.llm_client.client
        if getattr(self.llm_client, "text_format_supported", True):
            try:
                response = client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=JudgeVerdict,
                    temperature=temperature,
                )
                if response.output_parsed is not None:
                    return response.output_parsed
            except Exception:
                pass
        # Some OpenAI-compatible servers ignore structured output and reply
        # with prose — salvage the verdict from the raw text instead of
        # failing the whole flow.
        response = client.responses.create(
            model=self.model, input=messages, temperature=temperature
        )
        return self._parse_judge_text(response.output_text)

    @staticmethod
    def _parse_judge_text(text):
        if not isinstance(text, str) or not text:
            return None
        cleaned = text.strip().strip("`")
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return JudgeVerdict.model_validate(data)
            except Exception:
                pass
        verdict = None
        confidence = None
        answer = None
        verdict_match = re.search(
            r'verdict\s*[:=]\s*"?\s*(answer_partial|answer|escalate|reject)',
            cleaned,
            re.IGNORECASE,
        ) or re.search(
            r"^\s*(answer_partial|answer|escalate|reject)\s*:",
            cleaned,
            re.MULTILINE | re.IGNORECASE,
        )
        if verdict_match:
            verdict = verdict_match.group(1).lower()
        confidence_match = re.search(
            r'confidence\s*[:=]\s*"?\s*([0-9]*\.?[0-9]+)', cleaned, re.IGNORECASE
        )
        if confidence_match:
            confidence = min(max(float(confidence_match.group(1)), 0.0), 1.0)
        if verdict in ("answer", "answer_partial"):
            answer_match = re.search(
                r"^\s*(?:answer_partial|answer)\s*:\s*(.+)$",
                cleaned,
                re.MULTILINE | re.DOTALL,
            )
            if not answer_match:
                answer_match = re.search(r'"answer"\s*:\s*(.+)$', cleaned, re.DOTALL)
            if answer_match:
                answer = re.sub(
                    r"\n\s*confidence\s*[:=].*$",
                    "",
                    answer_match.group(1),
                    flags=re.MULTILINE,
                ).strip()
        try:
            return JudgeVerdict(
                verdict=verdict or "escalate",
                confidence=confidence if confidence is not None else 0.0,
                answer=answer,
            )
        except Exception:
            return None

    @staticmethod
    def _format_context(results, source_label):
        blocks = []
        for item in results:
            text = item.get("search_text") or item.get("snippet") or ""
            if text:
                blocks.append(text)
        body = "\n\n".join(blocks) if blocks else "(no results)"
        return f"Source: {source_label}\n\n{body}"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, query):
        initial = {"query": query}
        # Bound the graph so a pathological routing path cannot hang the UI.
        state = self.graph.invoke(
            initial,
            config={"recursion_limit": max(10, self.max_iterations * 5)},
        )
        return {
            "answer": state.get("answer") or "",
            "searches": state.get("searches", []),
            "iterations": state.get("iterations", 0),
            "rejected": state.get("rejected", False),
            "source": state.get("source"),
            "confidence": state.get("confidence"),
        }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = RAGAgent()
    result = agent.run("What are Pikachu's stats?")
    print(f"Answer: {result['answer'][:200]}")
    print(f"Source: {result['source']}")
    print(f"Iterations: {result['iterations']}")
    for i, s in enumerate(result["searches"]):
        print(f"  Search {i+1}: query='{s.query}', source={s.source}, results={len(s.results)}")

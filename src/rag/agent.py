"""Agentic RAG pipeline with iterative search and query reformulation.

Implements the agent loop pattern from Module 1 (1-Agentic RAG):
1. Perform initial hybrid search
2. Analyze results quality with LLM
3. Reformulate query if results are insufficient
4. Repeat up to MAX_ITERATIONS times
5. Generate final answer with all gathered context
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.llm import get_model
from src.rag.pipeline import RAGBase
from src.search.hybrid import HybridSearch

# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

ANALYSIS_INSTRUCTIONS = """\
You are evaluating search results for a Pokémon question-answering system.

Given a question and a set of search results (documents), determine:
1. Whether the question is about the Pokémon domain (stats, types,
   weaknesses, abilities, evolutions, team building). If it is about anything
   else — battle simulation, winner prediction, save files, cheating, or
   unrelated topics (cooking, finance, medicine, software, etc.) — set
   off_topic to true.
2. Whether the results contain enough information to answer the question confidently.
3. If not, suggest a reformulated query that might find better results.

Respond with a JSON object:
{
  "sufficient": true/false,
  "reason": "brief explanation",
  "reformulated_query": "improved search query (only if sufficient=false)",
  "off_topic": false,
  "off_topic_reason": ""
}

If results are sufficient, set reformulated_query to an empty string.
If the question is about Pokémon, set off_topic to false.
"""

ANSWER_INSTRUCTIONS = """\
You are a Pokémon knowledge assistant.

Use ONLY the information in the context to answer. Be concise and accurate.
- For weakness or resistance questions, cite the damage_taken multipliers from
  the retrieved documents (e.g. "Charizard is 4x weak to Rock, 2x weak to
  Water/Electric, and immune to Ground").
- For team-building questions, suggest Pokémon or types that cover the team's
  weaknesses based on the retrieved type data. Never simulate battles, predict
  winners, or claim that a team "will beat" another.
If the answer is not found in the context, say "I don't know."
If multiple searches were performed, synthesize information from all results.
"""

MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

REJECTION_MESSAGE = (
    "I'm a Pokémon knowledge assistant — I can answer questions about Pokémon "
    "stats, types, weaknesses, abilities, evolutions, and team building. I can't "
    "simulate battles, predict winners, access save files, or help with cheating. "
    "Try asking about a specific Pokémon!"
)

# In-domain low-confidence note (rejected: false — never a rejection dict).
UNCERTAINTY_NOTE = (
    "I couldn't find a confident answer to that in the Pokédex. "
    "Could you rephrase, or ask about a specific Pokémon?"
)

# Rule pre-gate: regex patterns matched against the lowercased query.
# Deliberately phrase/word-based so bare "battle" (battle TEAM suggestions,
# in-scope) never trips the gate.
REJECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Battle simulation / outcome prediction
    re.compile(r"who would win"),
    re.compile(r"predict (?:the )?winner"),
    re.compile(r"battle simulation"),
    re.compile(r"battle sim"),
    re.compile(r"simulate battle"),
    re.compile(r"battle outcome"),
    re.compile(r"win rate"),
    # Save files
    re.compile(r"save ?file"),
    re.compile(r"save ?game"),
    re.compile(r"\.sav\b"),
    # Cheating / hacking / emulation
    re.compile(r"\bcheat"),
    re.compile(r"\bhack"),
    re.compile(r"\bemulator"),
    re.compile(r"\bshowdown"),
    # Non-Pokémon topics
    re.compile(r"\bdocker\b"),
    re.compile(r"\bcourse\b"),
    re.compile(r"\bpython\b"),
    re.compile(r"\bcook"),
    re.compile(r"\bfinance"),
    re.compile(r"\bstock\b"),
    re.compile(r"\binvest"),
    re.compile(r"\bmedic"),
    re.compile(r"\bfever\b"),
)

# Deterministic fail-safe domain signals (static only, no data/ dependency):
# the 18 Pokémon type names plus high-confidence Pokédex vocabulary. Type-name
# substrings deliberately also catch compound Pokémon names (dragonite, darkrai).
POKEMON_TYPE_NAMES = frozenset({
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "steel", "dark", "fairy",
})

# Dex numbers 1..1025 (whole numbers only) and the "stat(s)" vocabulary term.
_DEX_NUMBER_RE = re.compile(r"\b(?:[1-9][0-9]{0,2}|10[0-2][0-5])\b")
_STAT_TERM_RE = re.compile(r"\bstat(?:s)?\b")


def rejection_result() -> dict:
    """Structured rejection dict for out-of-scope queries."""
    return {
        "answer": REJECTION_MESSAGE,
        "searches": [],
        "iterations": 0,
        "rejected": True,
    }


def _is_out_of_scope(query: str) -> bool:
    """Rule pre-gate: True when the query matches a rejection pattern."""
    normalized = query.lower()
    return any(pattern.search(normalized) for pattern in REJECT_PATTERNS)


def _has_pokemon_signals(text: str) -> bool:
    """True when the text carries static Pokémon-domain signals.

    Static only (no data/ dependency): dex numbers, the 18 type names,
    "pokemon"/"pokémon", and the "stat(s)" vocabulary term.
    """
    normalized = text.lower()
    if "pokemon" in normalized or "pokémon" in normalized:
        return True
    if _DEX_NUMBER_RE.search(normalized):
        return True
    if _STAT_TERM_RE.search(normalized):
        return True
    return any(t in normalized for t in POKEMON_TYPE_NAMES)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class SearchRecord:
    """One search iteration's results."""
    query: str
    results: list[dict]
    analysis: dict[str, Any] | None = None


@dataclass
class AgentResult:
    """Final output from the agent loop."""
    answer: str
    searches: list[SearchRecord]
    iterations: int


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

class RAGAgent:
    """Agentic RAG: iterative search with LLM-driven query reformulation.

    Demonstrates Module 1 patterns (agent loop, tool calling) applied to
    a RAG pipeline. The agent decides whether search results are sufficient
    and reformulates queries when they aren't.
    """

    def __init__(
        self,
        search_index: HybridSearch | None = None,
        llm_client: OpenAI | None = None,
        model: str | None = None,
        max_iterations: int = MAX_ITERATIONS,
        search_type: str = "hybrid",
    ):
        if search_index is None:
            search_index = HybridSearch()
        model = model or get_model()
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

    # ------------------------------------------------------------------
    # Tool: perform_search
    # ------------------------------------------------------------------

    def perform_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Execute hybrid search for the given query."""
        return self.rag.search(query, num_results=num_results)

    # ------------------------------------------------------------------
    # Tool: analyze_results
    # ------------------------------------------------------------------

    def analyze_results(self, query: str, results: list[dict]) -> dict:
        """Ask the LLM to evaluate whether search results are sufficient."""
        context = self.rag.build_context(results)
        prompt = (
            f"Question: {query}\n\n"
            f"Search Results:\n{context}\n\n"
            "Evaluate whether these results are sufficient to answer the question."
        )

        messages = [
            {"role": "developer", "content": ANALYSIS_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=messages,
        )

        text = response.output_text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            text = text.removeprefix("json")
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Deterministic fail-safe: never blindly treat unparseable analysis
            # as sufficient. If the query carries static Pokémon-domain signals,
            # keep the current fail-open behavior; otherwise reject the query as
            # out-of-scope by default.
            if _has_pokemon_signals(query):
                return {
                    "sufficient": True,
                    "reason": "Could not parse analysis (fallback)",
                    "reformulated_query": "",
                    "off_topic": False,
                    "off_topic_reason": "",
                }
            return rejection_result()

    # ------------------------------------------------------------------
    # Tool: reformulate_query
    # ------------------------------------------------------------------

    def reformulate_query(self, original_query: str, analysis: dict) -> str:
        """Use the LLM to reformulate the query for better results."""
        reformulated = analysis.get("reformulated_query", "").strip()
        if reformulated:
            return reformulated

        # If analysis didn't provide a reformulation, ask the LLM directly
        prompt = (
            f"Original query: {original_query}\n"
            f"Reason results were insufficient: {analysis.get('reason', 'unknown')}\n\n"
            "Provide a reformulated search query that would find better results. "
            "Return ONLY the new query, nothing else."
        )

        messages = [
            {"role": "developer", "content": "You are a search query reformulation assistant. Return only the reformulated query."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=messages,
        )

        return response.output_text.strip()

    # ------------------------------------------------------------------
    # Tool: generate_answer
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_results(all_results: list[dict]) -> list[dict]:
        """Deduplicate results by id, preserving first-seen order."""
        seen_ids: set[str] = set()
        unique_results: list[dict] = []
        for doc in all_results:
            doc_id = doc.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)
        return unique_results

    def generate_answer(self, query: str, all_results: list[dict]) -> str:
        """Generate the final answer using all gathered search results."""
        unique_results = self._dedupe_results(all_results)

        prompt = self.rag.build_prompt(query, unique_results)
        messages = [
            {"role": "developer", "content": ANSWER_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=messages,
        )

        return response.output_text

    # ------------------------------------------------------------------
    # Agent loop: run
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict:
        """Execute the agentic RAG loop.

        1. Reject out-of-scope queries up front (rule pre-gate)
        2. Search with the query
        3. Analyze if results are sufficient
        4. If not, reformulate and search again
        5. Repeat up to max_iterations
        6. Generate final answer from all collected results

        Returns:
            Dict with 'answer' (str), 'searches' (list of SearchRecord),
            and 'iterations' (int). Out-of-scope queries return a structured
            rejection dict with 'rejected' (True), empty searches, and 0
            iterations. In-domain low-confidence queries (empty results,
            exhausted iterations with all analyses insufficient, or an empty
            final context) return the uncertainty note with 'rejected' (False).
        """
        # Guardrail (layer 1): rule pre-gate rejects out-of-scope queries
        # before any search is performed.
        if _is_out_of_scope(query):
            return rejection_result()

        searches: list[SearchRecord] = []
        all_results: list[dict] = []
        current_query = query

        for i in range(self.max_iterations):
            # Step 1: Perform search
            results = self.perform_search(current_query)
            all_results.extend(results)

            # Step 2: Analyze results
            analysis = self.analyze_results(current_query, results)

            # Guardrail (layer 3): the deterministic fail-safe returned a
            # rejection dict when the analysis could not be parsed and the
            # query carried no Pokémon-domain signals.
            if analysis.get("rejected", False):
                return analysis

            # Guardrail (layer 2): the analyzer flagged the query as off-topic.
            if analysis.get("off_topic", False):
                return rejection_result()

            record = SearchRecord(
                query=current_query,
                results=results,
                analysis=analysis,
            )
            searches.append(record)

            # Step 3: Check if sufficient
            if analysis.get("sufficient", True):
                break

            # Step 4: Reformulate query for next iteration
            current_query = self.reformulate_query(query, analysis)

        # Step 5: Low-confidence path — in-domain queries the loop could not
        # answer confidently get the uncertainty note with rejected: false
        # (never a rejection dict). Triggers: zero results across all
        # searches (no LLM answer call at all), max_iterations reached with
        # every analysis insufficient, or an empty final context.
        unique_results = self._dedupe_results(all_results)
        final_context = self.rag.build_context(unique_results)
        exhausted = (
            len(searches) >= self.max_iterations
            and all(not s.analysis.get("sufficient", True) for s in searches)
        )
        if not all_results or not final_context.strip() or exhausted:
            return {
                "answer": UNCERTAINTY_NOTE,
                "searches": searches,
                "iterations": len(searches),
                "rejected": False,
            }

        # Step 6: Generate final answer
        answer = self.generate_answer(query, all_results)

        return {
            "answer": answer,
            "searches": searches,
            "iterations": len(searches),
            "rejected": False,
        }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = RAGAgent()
    result = agent.run("What is machine learning?")
    print(f"Answer: {result['answer'][:200]}")
    print(f"Iterations: {result['iterations']}")
    for i, s in enumerate(result["searches"]):
        print(f"  Search {i+1}: query='{s.query}', results={len(s.results)}, sufficient={s.analysis.get('sufficient') if s.analysis else 'N/A'}")

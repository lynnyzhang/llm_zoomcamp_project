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
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from src.llm import get_model
from src.rag.pipeline import RAGBase
from src.search.hybrid import HybridSearch


# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

ANALYSIS_INSTRUCTIONS = """\
You are evaluating search results for a question-answering system.

Given a question and a set of search results (documents), determine:
1. Whether the results contain enough information to answer the question confidently.
2. If not, suggest a reformulated query that might find better results.

Respond with a JSON object:
{
  "sufficient": true/false,
  "reason": "brief explanation",
  "reformulated_query": "improved search query (only if sufficient=false)"
}

If results are sufficient, set reformulated_query to an empty string.
"""

ANSWER_INSTRUCTIONS = """\
You are a helpful assistant that answers questions based on provided context.

Use ONLY the information in the context to answer. Be concise and accurate.
If the answer is not found in the context, say "I don't know."
If multiple searches were performed, synthesize information from all results.
"""

MAX_ITERATIONS = 3


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
    ):
        if search_index is None:
            search_index = HybridSearch()
        model = model or get_model()
        self.rag = RAGBase(
            search_index=search_index,
            llm_client=llm_client,
            model=model,
        )
        self.llm_client = self.rag.llm_client
        self.model = model
        self.max_iterations = max_iterations

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
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: treat as sufficient if we can't parse
            return {"sufficient": True, "reason": "Could not parse analysis", "reformulated_query": ""}

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

    def generate_answer(self, query: str, all_results: list[dict]) -> str:
        """Generate the final answer using all gathered search results."""
        # Deduplicate results by id
        seen_ids: set[str] = set()
        unique_results: list[dict] = []
        for doc in all_results:
            doc_id = doc.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)

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

        1. Search with the query
        2. Analyze if results are sufficient
        3. If not, reformulate and search again
        4. Repeat up to max_iterations
        5. Generate final answer from all collected results

        Returns:
            Dict with 'answer' (str), 'searches' (list of SearchRecord),
            and 'iterations' (int).
        """
        searches: list[SearchRecord] = []
        all_results: list[dict] = []
        current_query = query

        for i in range(self.max_iterations):
            # Step 1: Perform search
            results = self.perform_search(current_query)
            all_results.extend(results)

            # Step 2: Analyze results
            analysis = self.analyze_results(current_query, results)

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

        # Step 5: Generate final answer
        answer = self.generate_answer(query, all_results)

        return {
            "answer": answer,
            "searches": searches,
            "iterations": len(searches),
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

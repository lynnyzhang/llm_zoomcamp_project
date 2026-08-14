import json
import logging
import os
import re
from dataclasses import dataclass

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
You are a Pokémon knowledge assistant that answers questions about the
1,350-Pokémon knowledge base and Bulbapedia. You decide yourself when to
search, using two tools:

- search_local_knowledge_base(query): the local knowledge base — stats,
  types, weaknesses, abilities, evolutions, alternate forms, type charts.
  Use this first for any Pokémon question.
- search_bulbapedia(query): web search of Bulbapedia for facts the local
  base lacks (moves, anime, manga, lore, game history, strategy). Pass a
  short keyword query with the Pokémon name and the missing facts.

Rules:
- Answer ONLY from retrieved tool results — never from memory.
- If the local results confidently answer the question, reply with the
  answer. If they are insufficient or only partial, call search_bulbapedia
  and answer with the combined results.
- A grounded partial answer is better than a refusal: state what the tools
  support, then state what they do not determine, and hedge ("based on the
  retrieved data", "the tools do not say"). You may decline to rank "best"
  picks while giving type-coverage guidance from the retrieved data.
- Out of scope — reply with the rejection message verbatim and do NOT call
  tools: predicting winners or simulating the outcome of a specific battle,
  access to save files, cheats, emulators, real-time data, or any
  non-Pokémon topic. Team-building and type-matchup questions ARE in scope.
- never guess: when the tools yield no confident answer, reply with the
  rejection message verbatim:
{REJECTION_MESSAGE}
"""

MAX_ITERATIONS = 3

LOCAL_SEARCH_TOOL = {
    "type": "function",
    "name": "search_local_knowledge_base",
    "description": (
        "Search the local Pokémon knowledge base (1,350 Pokémon: stats, "
        "types, weaknesses, abilities, evolutions, alternate forms, type "
        "charts). Use this first for any Pokémon question."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "search_bulbapedia",
    "description": (
        "Search Bulbapedia for Pokémon facts the local knowledge base lacks "
        "(moves, anime, manga, lore, game history, strategy). Pass a short "
        "keyword query with the Pokémon name and the missing facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Short keyword query"}
        },
        "required": ["query"],
    },
}

TOOLS = [LOCAL_SEARCH_TOOL, WEB_SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Agent state / records
# ---------------------------------------------------------------------------


@dataclass
class SearchRecord:
    query: str
    results: list[dict]
    source: str | None = None
    search_query: str | None = None


def get_confidence_threshold():
    return float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))


def cosine_similarity(embedder, text_a, text_b):
    """Cosine similarity of two texts' embedding vectors (0..1).

    Used for the two RAG quality scores: grounding (answer vs retrieved
    context — faithfulness) and relevance (question vs answer)."""
    if not text_a or not text_b:
        return 0.0
    va = embedder.encode(text_a, normalize=False)
    vb = embedder.encode(text_b, normalize=False)
    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = sum(x * x for x in va) ** 0.5
    norm_b = sum(y * y for y in vb) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# RAGAgent (manual tool-use loop)
# ---------------------------------------------------------------------------


class RAGAgent:

    def __init__(
        self,
        search_index=None,
        llm_client=None,
        model=None,
        max_iterations=MAX_ITERATIONS,
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
        )
        self.llm_client = self.rag.llm_client
        self.model = model
        self.max_iterations = max_iterations
        self.num_results = num_results
        # Reuses the search index's embedder for the grounding/relevance
        # scores; without one (stub indexes) the scores are skipped.
        self.embedder = getattr(search_index, "embedder", None)
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_confidence_threshold()
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def perform_search(self, query, num_results=5):
        return self.rag.search(query, num_results=num_results)

    def execute_tool(self, name, arguments, question):
        if name == "search_local_knowledge_base":
            results = self.perform_search(
                arguments.get("query", question), num_results=self.num_results
            )
            record = SearchRecord(query=question, results=results, source="local")
            return record, self.format_tool_results(results)
        if name == "search_bulbapedia":
            query = arguments.get("query", question)
            try:
                results = web.web_search(query, num_results=self.num_results)
            except Exception:
                # Broad except: any web failure (missing/invalid key, usage
                # limit, network) yields empty results so the model still
                # gets a tool response and can decide without crashing.
                logging.getLogger(__name__).warning(
                    "Web search failed for %r", question, exc_info=True
                )
                results = []
            record = SearchRecord(
                query=question,
                results=results,
                source="web",
                search_query=query,
            )
            return record, self.format_tool_results(results)
        return None, json.dumps({"error": f"unknown tool: {name}"})

    @staticmethod
    def format_tool_results(results):
        blocks = []
        for item in results:
            if "snippet" in item:
                blocks.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": (item.get("snippet") or "")[:800],
                        "score": round(float(item.get("score") or 0.0), 3),
                    }
                )
            else:
                blocks.append(
                    {
                        "name": item.get("name", ""),
                        "text": (item.get("search_text") or "")[:1500],
                    }
                )
        return json.dumps(blocks, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, query):
        # Deterministic guard: empty/whitespace or pure punctuation can never
        # be a question — reject without spending an LLM round trip.
        if not query or not query.strip() or not re.search(r"[a-zA-Z0-9]", query):
            return {
                "answer": REJECTION_MESSAGE,
                "searches": [],
                "iterations": 0,
                "rejected": True,
                "source": None,
                "confidence": None,
                "relevance": None,
            }

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
                response = self.llm_client.client.responses.create(
                    model=self.model,
                    input=items,
                    tools=tools,
                    temperature=LLMClient.get_agent_temperature(),
                )
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    return self.finalize(response.output_text, query, searches, sources)
                items += calls
                for call in calls:
                    try:
                        arguments = json.loads(call.arguments or "{}")
                    except Exception:
                        arguments = {}
                    record, output = self.execute_tool(call.name, arguments, query)
                    if record is not None:
                        searches.append(record)
                        sources.add(record.source or "")
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": getattr(call, "call_id", None)
                            or f"call_{len(searches)}",
                            "output": output,
                        }
                    )
        except Exception:
            # Fail-soft: an LLM/network failure must never fabricate an answer.
            logging.getLogger(__name__).warning(
                "Agent loop failed for %r", query, exc_info=True
            )
        # Loop exhausted without a final answer: never guess.
        return self.finalize(None, query, searches, sources)

    def finalize(self, answer_text, query, searches, sources):
        answer = (answer_text or "").strip()
        if not answer or answer == REJECTION_MESSAGE:
            return {
                "answer": REJECTION_MESSAGE,
                "searches": searches,
                "iterations": len(searches),
                "rejected": True,
                "source": None,
                "confidence": None,
                "relevance": None,
            }
        confidence = None
        relevance = None
        if self.embedder is not None:
            # Grounding = the best match against any single retrieved
            # record — concatenating all records dilutes the answer's
            # actual source. Relevance = the question vs the answer.
            scores = []
            for record in searches:
                for item in record.results:
                    text = item.get("search_text") or item.get("snippet") or ""
                    if text:
                        scores.append(cosine_similarity(self.embedder, answer, text))
            confidence = max(scores) if scores else 0.0
            relevance = cosine_similarity(self.embedder, query, answer)
        if confidence is not None and confidence < self.confidence_threshold:
            # Ungrounded answers (memory, invented facts, tool-less replies)
            # fail the gate — never surface them.
            return {
                "answer": REJECTION_MESSAGE,
                "searches": searches,
                "iterations": len(searches),
                "rejected": True,
                "source": None,
                "confidence": None,
                "relevance": None,
            }
        return {
            "answer": answer,
            "searches": searches,
            "iterations": len(searches),
            "rejected": False,
            "source": "web" if "web" in sources else ("local" if "local" in sources else None),
            "confidence": confidence,
            "relevance": relevance,
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

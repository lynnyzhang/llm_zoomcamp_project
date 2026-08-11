import json
import re
from dataclasses import dataclass
from typing import Any

from src.llm import get_model
from src.rag.pipeline import RAGBase
from src.search.hybrid import HybridSearch
from src.search.web import web_search

# ---------------------------------------------------------------------------
# Agent instructions
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

# Tool-use developer instructions (reference-style: "make multiple searches
# with different keywords before answering") + the Pokémon persona and
# guardrail clauses from the previous analysis-based loop.
AGENT_INSTRUCTIONS = f"""\
You are a Pokémon knowledge assistant. Use the search tool to find
information in the local Pokédex index, and the web_search tool when the
question needs information beyond the index (recent news, events, or topics
not in the dataset). Make multiple searches with different keywords before
answering.

Rules:
- Answer ONLY from the information returned by the tools.
- For weakness or resistance questions, cite the damage_taken multipliers from
  the retrieved documents (e.g. "Charizard is 4x weak to Rock, 2x weak to
  Water/Electric, and immune to Ground").
- For team-building questions, suggest Pokémon or types that cover the team's
  weaknesses based on the retrieved type data. Never simulate battles, predict
  winners, or claim that a team "will beat" another.
- If the answer is not found in the tool results, say "I don't know."
- If the question is outside the Pokémon domain (battle simulation, winner
  prediction, save files, cheating, or unrelated topics such as cooking,
  finance, medicine, software), do NOT call any tool. Instead, reply with
  exactly: {REJECTION_MESSAGE}
"""

MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Tools (reference-style function tools, 1-Agentic RAG)
# ---------------------------------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": "Search the local Pokémon Pokédex index (hybrid keyword + "
                   "vector search) for documents matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the Pokédex index.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web (DuckDuckGo) for information not covered by "
                   "the local Pokédex index — e.g. recent events, news, or "
                   "topics outside the dataset.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Web search query text.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

AGENT_TOOLS = [SEARCH_TOOL, WEB_SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

# Rule pre-gate: regex patterns matched against the lowercased query.
# Deliberately phrase/word-based so bare "battle" (battle TEAM suggestions,
# in-scope) never trips the gate.
REJECT_PATTERNS = (
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


def _is_out_of_scope(query):
    normalized = query.lower()
    return any(pattern.search(normalized) for pattern in REJECT_PATTERNS)


def rejection_result():
    return {
        "answer": REJECTION_MESSAGE,
        "searches": [],
        "iterations": 0,
        "rejected": True,
    }


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class SearchRecord:
    query: str
    results: list[dict]
    analysis: dict[str, Any] | None = None


@dataclass
class AgentResult:
    answer: str
    searches: list[SearchRecord]
    iterations: int


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

class RAGAgent:

    def __init__(
        self,
        search_index=None,
        llm_client=None,
        model=None,
        max_iterations=MAX_ITERATIONS,
        search_type="hybrid",
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
    # Tool: search
    # ------------------------------------------------------------------

    def perform_search(self, query, num_results=5):
        return self.rag.search(query, num_results=num_results)

    # ------------------------------------------------------------------
    # Tool dispatch (reference make_call)
    # ------------------------------------------------------------------

    def _make_call(self, call, searches):
        """Execute a function_call item: dispatch on the tool name and return
        the function_call_output message to feed back into the conversation
        (reference pattern: execute → JSON result → append to messages)."""
        try:
            args = json.loads(call.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        query = (args.get("query") or "").strip()

        if not query:
            # Fail-safe: never run a search on unparseable/empty arguments —
            # feed the error back so the model can recover.
            output = json.dumps({"error": f"Could not parse arguments for tool '{call.name}': missing 'query'."})
        elif call.name == "search":
            results = self.perform_search(query, num_results=5)
            searches.append(
                SearchRecord(query=query, results=results, analysis={"sufficient": len(results) > 0})
            )
            output = json.dumps(results, ensure_ascii=False)
        elif call.name == "web_search":
            results = web_search(query, num_results=5)
            searches.append(
                SearchRecord(query=query, results=results, analysis={"sufficient": len(results) > 0})
            )
            output = json.dumps(results, ensure_ascii=False)
        else:
            output = json.dumps({"error": f"Unknown tool: {call.name}"})

        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": output,
        }

    # ------------------------------------------------------------------
    # Agent loop: run (reference tool-use loop, bounded)
    # ------------------------------------------------------------------

    def run(self, query):
        # Guardrail (layer 1): rule pre-gate rejects out-of-scope queries
        # before any LLM call or search is performed.
        if _is_out_of_scope(query):
            return rejection_result()

        messages = [
            {"role": "developer", "content": AGENT_INSTRUCTIONS},
            {"role": "user", "content": query},
        ]
        searches = []
        last_answer = UNCERTAINTY_NOTE

        # Reference-style loop: the model decides when to search (function
        # calls) and when to answer (a plain message — the loop stops when a
        # response contains no function calls). Bounded by max_iterations so
        # a looping model cannot hang the UI.
        for _ in range(self.max_iterations):
            response = self.llm_client.responses.create(
                model=self.model,
                input=messages,
                tools=AGENT_TOOLS,
            )
            messages.extend(response.output)
            called_tool = False
            for item in response.output:
                if item.type == "function_call":
                    messages.append(self._make_call(item, searches))
                    called_tool = True
                elif item.type == "message":
                    last_answer = item.content[0].text
            if not called_tool:
                break

        # Guardrail (layer 2): LLM-level off-topic — the model was instructed
        # to refuse out-of-domain questions with the exact rejection message
        # and no tool calls; surface it through the same rejection contract.
        if not searches and last_answer == REJECTION_MESSAGE:
            return rejection_result()

        return {
            "answer": last_answer,
            "searches": searches,
            "iterations": len(searches),
            "rejected": False,
        }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = RAGAgent()
    result = agent.run("What are Pikachu's stats?")
    print(f"Answer: {result['answer'][:200]}")
    print(f"Iterations: {result['iterations']}")
    for i, s in enumerate(result["searches"]):
        print(f"  Search {i+1}: query='{s.query}', results={len(s.results)}")

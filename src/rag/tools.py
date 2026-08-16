import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.search import web_search
from src.search.search_records import SearchResult, WebResult

if TYPE_CHECKING:
    from src.rag.rag_agent import RAGAgent

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


@dataclass
class SearchRecord:
    query: str
    results: Sequence[SearchResult | WebResult]
    source: str | None = None
    search_query: str | None = None

    @property
    def payload(self):
        # Compact per-result payload for the searches store: ids/scores for
        # local docs, title/url/snippet for web items, snippet truncated to
        # 300 chars.
        items = []
        for item in self.results:
            if isinstance(item, WebResult):
                items.append(
                    {
                        "title": item.title,
                        "url": item.url,
                        "snippet": (item.snippet or "")[:300],
                        "score": item.score,
                    }
                )
            else:
                items.append(
                    {
                        "id": item.id,
                        "name": getattr(item, "name", ""),
                        "score": item.score,
                    }
                )
        return items


def format_tool_results(results: Sequence[SearchResult | WebResult]) -> str:
    blocks = []
    for item in results:
        if isinstance(item, WebResult):
            blocks.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": (item.snippet or "")[:800],
                    "score": round(float(item.score or 0.0), 3),
                }
            )
        else:
            blocks.append(
                {
                    "name": getattr(item, "name", ""),
                    "text": (item.search_text or "")[:1500],
                }
            )
    return json.dumps(blocks, ensure_ascii=False)


def execute_tool(
    agent: "RAGAgent", name: str, arguments: dict, question: str
) -> tuple[SearchRecord | None, str]:
    if name == "search_local_knowledge_base":
        query = arguments.get("query", question)
        results = agent.search(query, num_results=agent.num_results)
        record = SearchRecord(
            query=question,
            results=results,
            source="local",
            search_query=query,
        )
        return record, format_tool_results(results)
    if name == "search_bulbapedia":
        query = arguments.get("query", question)
        try:
            results = web_search.web_search(query, num_results=agent.num_results)
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
        return record, format_tool_results(results)
    return None, json.dumps({"error": f"unknown tool: {name}"})


def apply_tool_calls(
    agent: "RAGAgent",
    calls: list,
    question: str,
    messages: list[dict],
    searches: list[SearchRecord],
    sources: set[str],
):
    messages += calls
    for call in calls:
        try:
            arguments = json.loads(call.arguments or "{}")
        except Exception:
            arguments = {}
        record, output = execute_tool(agent, call.name, arguments, question)
        if record is not None:
            searches.append(record)
            sources.add(record.source or "")
        messages.append(
            {
                "type": "function_call_output",
                "call_id": getattr(call, "call_id", None) or f"call_{len(searches)}",
                "output": output,
            }
        )

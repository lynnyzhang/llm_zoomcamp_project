from dataclasses import dataclass

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
    results: list[dict]
    source: str | None = None
    search_query: str | None = None

    @property
    def payload(self):
        # Compact per-result payload for the searches store: ids/scores for
        # local docs, title/url/snippet for web items, snippet truncated to
        # 300 chars.
        items = []
        for item in self.results:
            if "snippet" in item:
                items.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": (item.get("snippet") or "")[:300],
                    "score": item.get("score"),
                })
            else:
                items.append({
                    "id": item.get("id"),
                    "name": item.get("name", ""),
                    "score": item.get("score"),
                })
        return items

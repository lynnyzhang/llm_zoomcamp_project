# src/search/web.py
#
# Web search backend for the agent's escalation path: Tavily, restricted to
# bulbapedia.bulbagarden.net. Returns a small list of {title, url, snippet}
# dicts (the same shape the old DuckDuckGo backend produced, so consumers are
# unchanged). The caller is responsible for handling failures (the agent's
# web_search node degrades gracefully to empty results).

import os

from tavily import TavilyClient


def get_api_key():
    key = os.environ.get("TAVILY_API_KEY")
    if key is None or not key.strip():
        raise RuntimeError(
            "TAVILY_API_KEY is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return key


def web_search(query, num_results=5, api_key=None):
    """Web search via Tavily, restricted to bulbapedia.bulbagarden.net.

    Returns up to num_results dicts {title, url, snippet}. Raises on missing
    key or API errors; the agent wraps this call so the flow degrades to a
    rejection instead of crashing.
    """
    client = TavilyClient(api_key=api_key or get_api_key())
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=num_results,
        include_domains=["bulbapedia.bulbagarden.net"],
    )
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        }
        for item in response.get("results", [])
        # Skip non-authoritative user subpages (User:, User talk:, blog
        # posts) — the judge must not ground answers on them.
        if "/user" not in item.get("url", "").lower()
    ]
    return results[:num_results]

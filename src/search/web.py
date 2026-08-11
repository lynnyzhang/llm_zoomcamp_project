# src/search/web.py
#
# Keyless web search backend for the agent's web_search tool: the DuckDuckGo
# Instant Answer API (free, no API key, via requests — already a project
# dependency). Returns a small list of {title, url, snippet} dicts and
# degrades gracefully to [] on any failure (offline, rate limits, timeouts)
# so the agent loop never breaks on web issues.

import requests

DDG_API_URL = "https://api.duckduckgo.com/"
TIMEOUT_SECONDS = 10


def _topic_to_result(topic):
    """Map a DuckDuckGo RelatedTopic (or nested Topics entry) to a result."""
    text = topic.get("Text") or ""
    return {
        "title": topic.get("Result") or (text.split(" - ")[0][:80] if text else ""),
        "url": topic.get("FirstURL") or "",
        "snippet": text,
    }


def web_search(query, num_results=5):
    """Web search via the DuckDuckGo Instant Answer API.

    Returns up to num_results dicts {title, url, snippet}. Never raises:
    any failure returns [] (the agent feeds empty results back to the LLM,
    which can then decide how to proceed).
    """
    try:
        response = requests.get(
            DDG_API_URL,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:  # noqa: BLE001 — web search must never break the agent loop
        return []

    results = []
    # The Instant Answer abstract is the top hit when one exists.
    abstract = data.get("AbstractText") or ""
    if abstract:
        results.append(
            {
                "title": data.get("Heading") or query,
                "url": data.get("AbstractURL") or "",
                "snippet": abstract,
            }
        )
    for topic in data.get("RelatedTopics") or []:
        if "Topics" in topic:  # category node → recurse one level
            for sub in topic.get("Topics") or []:
                results.append(_topic_to_result(sub))
        else:
            results.append(_topic_to_result(topic))

    # Deduplicate by URL (fall back to snippet prefix when no URL).
    seen = set()
    unique = []
    for r in results:
        key = r["url"] or r["snippet"][:40]
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:num_results]

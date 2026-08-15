import json
import logging

from src.search import web

from src.rag.tools import SearchRecord


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


def execute_tool(agent, name, arguments, question):
    if name == "search_local_knowledge_base":
        results = agent.perform_search(
            arguments.get("query", question), num_results=agent.num_results
        )
        record = SearchRecord(query=question, results=results, source="local")
        return record, format_tool_results(results)
    if name == "search_bulbapedia":
        query = arguments.get("query", question)
        try:
            results = web.web_search(query, num_results=agent.num_results)
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


def apply_tool_calls(agent, calls, question, items, searches, sources):
    items += calls
    for call in calls:
        try:
            arguments = json.loads(call.arguments or "{}")
        except Exception:
            arguments = {}
        record, output = execute_tool(agent, call.name, arguments, question)
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

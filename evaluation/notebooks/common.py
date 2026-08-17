"""Shared infrastructure for evaluation notebooks.

Module-level lazy setup: importing this module builds no agent and performs no
IO. Each notebook calls setup()/build_agent() explicitly.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.llm_client import LLMClient
from src.rag.rag_agent import RAGAgent
from src.search.hybrid_search import HybridSearch


def project_root() -> Path:
    """The repo root: climb from cwd until a directory containing src/."""
    root = Path.cwd()
    while not (root / "src").exists():
        if root.parent == root:
            break
        root = root.parent
    return root


def setup() -> Path:
    """Make src importable and load .env; returns the project root."""
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    return root


def build_agent():
    """Build the agent and return it with the real bound call_llm captured
    before any tracing: each trace_run must wrap the real method; wrapping the
    previous wrapper leaks calls into earlier logs."""
    agent = RAGAgent(search_index=HybridSearch(), llm_client=LLMClient.get())
    real_call_llm = agent.call_llm
    return agent, real_call_llm


def snapshot_request(messages) -> list:
    """JSON-safe per-message snapshot for request bodies."""
    snap = []
    for m in messages:
        if isinstance(m, dict):
            snap.append({k: v for k, v in m.items()})
        else:
            snap.append(
                {
                    "type": "function_call",
                    "call_id": getattr(m, "call_id", None),
                    "name": getattr(m, "name", None),
                    "arguments": getattr(m, "arguments", None),
                }
            )
    return snap


def trace_run(agent, query, real_call_llm):
    """Wrap real_call_llm for one run; returns result, per-call log, whether
    escalation fired, and the full gate_history."""
    from src.rag.prompts import ESCALATION_MESSAGE

    calls_log = []

    def traced(messages, tools=None, temperature=None):
        r = real_call_llm(messages, tools=tools, temperature=temperature)
        items = []
        for item in r.output:
            if item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except Exception:
                    args = {}
                items.append(
                    {
                        "type": "local"
                        if item.name == "search_local_knowledge_base"
                        else "web",
                        "name": item.name,
                        "search_query": args.get("query", ""),
                    }
                )
        escalated_now = any(
            isinstance(m, dict) and m.get("content") == ESCALATION_MESSAGE
            for m in messages
        )
        calls_log.append(
            {
                "items": items,
                "escalated": escalated_now,
                "request": snapshot_request(messages),
            }
        )
        return r

    agent.call_llm = traced
    try:
        result = agent.run(query)
    finally:
        agent.call_llm = real_call_llm
    return result, calls_log, any(c["escalated"] for c in calls_log), agent.gate_history


def load_qa(path) -> list:
    """Load the QA jsonl lines ({question, document})."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_qa(questions, n) -> list:
    """Deterministic first-n sample of the question list."""
    return questions[:n]

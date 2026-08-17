import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(
    md("""# 01 Agent Path Analysis — detailed Q→answer trace

Purpose: inspect the actual gate behavior on a mixed question set. For each
question this notebook records the **entire LLM call sequence** (request +
response bodies per call) and, for **each gate decision**, the confidence,
relevance, the answer, and the **exact retrieval text** used to compute that
confidence — with the pre-escalation gate flagged. The full trace is written to
`evaluation/notebooks/data/agent_path_trace.txt`.

Input: 6 local + 2 web + 2 partial + 1 multi-Pokémon question (the multi one
tests whether the LLM searches once per Pokémon).

Run cells top to bottom. ~11 questions × ~35s + index build ≈ ~8-10 min.""")
)

cells.append(
    md("""## 1. Setup

Builds the hybrid search index and the agent from `.env` config, via
`evaluation.notebooks.common`.""")
)

cells.append(
    code("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.notebooks.common import build_agent, trace_run, load_qa, sample_qa, setup

setup()
agent, real_call_llm = build_agent()
print("confidence threshold:", agent.confidence_threshold)
print("model:", agent.model)""")
)

cells.append(
    md("""## 2. Question set

Mixed set: the first 6 **local** questions of the dev-subset QA set, 2 **web**
(Bulbapedia-only), 2 **local-partial** (local answers most, web may add), and 1
**multi** (needs facts about 3 Pokémon — does the LLM search once per
Pokémon?).""")
)

cells.append(
    code("""QA = load_qa(PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl")
LOCAL = [
    {"question": q["question"], "nature": "local", "expected": "hybrid -> answer"}
    for q in sample_qa(QA, 6)
]
WEB = [
    {"question": "Who voiced Pikachu in the anime?", "nature": "web", "expected": "hybrid -> web -> answer"},
    {"question": "What is the newest Pokémon introduced in Scarlet/Violet?", "nature": "web", "expected": "hybrid -> web -> answer"},
]
PARTIAL = [
    {"question": "Since Ivysaur has the Overgrow ability, what happens if its HP gets really low in battle?", "nature": "local-partial", "expected": "hybrid -> answer"},
    {"question": "If I want to evolve Ivysaur into Venusaur, do I need any specific items or just level it up?", "nature": "local-partial", "expected": "hybrid -> answer"},
]
MULTI = [
    {"question": "Compare the base stat totals of Bulbasaur, Charmander, and Squirtle — which is highest and how do their speeds differ?", "nature": "multi", "expected": "hybrid -> answer"},
]
QUESTIONS = LOCAL + WEB + PARTIAL + MULTI
for i, q in enumerate(QUESTIONS, 1):
    print(f"[{i}] ({q['nature']}) {q['question']}")""")
)

cells.append(
    md("""## 3. Run

For each question, run the agent via `trace_run` (which now captures the full
request **and response** body per LLM call) and collect `gate_history` — every
grounding-gate decision with its confidence, relevance, answer, and the
retrieval text it was computed from.""")
)

cells.append(
    code("""trace_results = {}

for i, q in enumerate(QUESTIONS, 1):
    result, calls, escalated, gate_history = trace_run(agent, q["question"], real_call_llm)
    trace_results[i] = {"question": q["question"], "nature": q["nature"], "result": result, "calls": calls, "gate_history": gate_history}
    status = "rejected" if result.rejected else "accepted"
    print(f"[{i}/{len(QUESTIONS)}] {status} | conf={result.confidence and round(result.confidence, 3)} | calls={len(calls)} | gates={len(gate_history)}")""")
)

cells.append(
    md("""## 4. Write the detailed trace

Writes a readable per-question block to `evaluation/notebooks/data/agent_path_trace.txt`:
per LLM call the full request + response bodies; per gate decision the
confidence, relevance, answer, and the exact retrieval `search_text` used to
compute that confidence; the pre-escalation gate (the rejected attempt that
triggered escalation) is flagged; and the final outcome.""")
)

cells.append(
    code("""import json

TRACE_FILE = PROJECT_ROOT / "evaluation" / "notebooks" / "data" / "agent_path_trace.txt"
TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)


def retrieval_text(gate):
    texts = []
    for rec in gate.searches:
        for item in rec.results:
            label = f"id={getattr(item, 'id', '?')} name={getattr(item, 'name', '?')}"
            text = getattr(item, "search_text", "") or getattr(item, "snippet", "")
            texts.append(f"[{label}] {text}")
    return texts


with open(TRACE_FILE, "w", encoding="utf-8") as f:
    for i, t in trace_results.items():
        result = t["result"]
        calls = t["calls"]
        gate_history = t["gate_history"]
        f.write(f"=== Question {i} ===\\n")
        f.write(f"Q: {t['question']}\\n")
        for j, c in enumerate(calls, 1):
            items = ", ".join(f"({it['type']}, query={it['search_query']!r})" for it in c["items"]) or "no tool call"
            esc = " [ESC]" if c["escalated"] else ""
            f.write(f"--- LLM call {j} ---\\n")
            f.write(f"  items: [{items}]{esc}\\n")
            f.write(f"  request: {json.dumps(c['request'], ensure_ascii=False, indent=2)}\\n")
            f.write(f"  response: {json.dumps(c['response'], ensure_ascii=False, indent=2)}\\n")
        for k, g in enumerate(gate_history, 1):
            gs = "rejected" if g.rejected else "accepted"
            pre_esc = g.rejected and (k < len(gate_history))
            f.write(f"--- Gate decision {k} ({gs}, conf={g.confidence and round(g.confidence, 3)}) ---\\n")
            f.write(f"  relevance: {g.relevance and round(g.relevance, 3)}\\n")
            answer = g.rejected_answer if g.rejected else g.answer
            f.write(f"  answer: {answer}\\n")
            f.write("--- Retrieval text for this gate ---\\n")
            for txt in retrieval_text(g):
                f.write(f"  {txt}\\n")
            if pre_esc:
                f.write(f">>> PRE-ESCALATION GATE (rejected, conf={g.confidence and round(g.confidence, 3)}) <<<\\n")
        status = "rejected" if result.rejected else "accepted"
        f.write(f"--- Final outcome: {status}, source={result.source}, confidence={result.confidence and round(result.confidence, 3)}, relevance={result.relevance and round(result.relevance, 3)} ---\\n")
        f.write(f"  answer: {result.answer}\\n")
        f.write("\\n")

print(f"wrote {len(trace_results)} questions to {TRACE_FILE}")""")
)

cells.append(
    md("""## 5. Summary

Minimal per-question outcome at a glance (the full detail is in the trace
file).""")
)

cells.append(
    code("""import pandas as pd

rows = []
for i, t in trace_results.items():
    r = t["result"]
    rows.append({
        "#": i,
        "nature": t["nature"],
        "question": t["question"],
        "accepted": not r.rejected,
        "source": r.source,
        "confidence": round(r.confidence, 3) if r.confidence else None,
        "relevance": round(r.relevance, 3) if r.relevance is not None else None,
        "llm calls": len(t["calls"]),
        "gates": len(t["gate_history"]),
    })
df = pd.DataFrame(rows)
df""")
)

nb.cells = cells
nbf.write(nb, "evaluation/notebooks/01_agent_path_analysis.ipynb")
print("wrote evaluation/notebooks/01_agent_path_analysis.ipynb")

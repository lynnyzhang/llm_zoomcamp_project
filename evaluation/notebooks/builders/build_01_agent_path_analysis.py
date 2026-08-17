import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(
    md("""# 01 Agent Path Analysis

Purpose: troubleshoot agent flow. For each question this notebook records the
**raw facts** of every LLM call — tool type + `search_query` per call, the
full request bodies, and the final answer — plus the `gate_history` (every
grounding-gate decision: accepted/rejected with confidence and relevance, and
the rejected answer when a gate failed). It then flags abnormal paths where
the actual path differs from the expected one.

Input: the dev-subset QA set (`evaluation/data/qa.jsonl`).

Run cells top to bottom. Each question takes ~30-40s (local LLM via the
proxy).""")
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
    md("""## 2. Tracing

`trace_run(agent, query, real_call_llm)` wraps the real `agent.call_llm` for a
single run and returns four things:

- `result` — the final `AgentResult`
- `calls_log` — per call: `items` (each function call the model made — `type`
  (`local` = search_local_knowledge_base, `web` = search_bulbapedia), `name`,
  `search_query`), whether the escalation message was in that call's request,
  and a JSON-safe request-body snapshot
- `escalated` — whether escalation fired at any point in the run
- `gate_history` — every grounding-gate decision in order (`rejected`,
  `confidence`, `relevance`, `rejected_answer`)

`real_call_llm` is the agent's real bound method, captured before any tracing;
each `trace_run` wraps that and restores it afterwards, so no run's calls leak
into another's logs.""")
)

cells.append(
    md("""## 3. Question set

`nature` classifies what the question *needs* (judgment, kept separate from the
raw trace): **local** (answerable from the local KB), **web** (purely
Bulbapedia), **guardrail** (must be rejected; excluded from the escalation
stats).

`expected` is minimal and unprocessed:
- `hybrid -> answer` — first LLM call returns only the hybrid-search tool call, next call returns the answer
- `hybrid -> web -> answer` — hybrid search, then the model calls web on its own, then answers
- `reject` — must be rejected""")
)

cells.append(
    code("""QA = load_qa(PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl")
SAMPLE = 20
QUESTIONS = [
    {"question": q["question"], "nature": "local", "expected": "hybrid -> answer"}
    for q in sample_qa(QA, SAMPLE)
]
QUESTIONS += [
    # --- web-only (Bulbapedia) ---
    {"question": "Who voiced Pikachu in the anime?", "nature": "web", "expected": "hybrid -> web -> answer"},
    {"question": "What is the newest Pokémon introduced in Scarlet/Violet?", "nature": "web", "expected": "hybrid -> web -> answer"},
    # --- guardrail (must reject; excluded from escalation analysis) ---
    {"question": "Who won the 2024 Super Bowl?", "nature": "guardrail", "expected": "reject"},
    {"question": "asdfghjkl", "nature": "guardrail", "expected": "reject"},
    {"question": "Tell me about Abraham Lincoln", "nature": "guardrail", "expected": "reject"},
]""")
)

cells.append(
    md("""## 4. Run

For each question, print the raw per-call record: call items (type +
`search_query` pairs), `[ESC]` when the escalation message was in that call's
request body, the request body of each call (roles/contents, tool outputs
truncated in this print — the full bodies stay in `trace_results`), the final
answer, and the gate-history block (every gate decision in order, which is
what makes gate-failure diagnosis possible).""")
)

cells.append(
    code("""trace_results = {}

for i, q in enumerate(QUESTIONS, 1):
    result, calls, escalated, gate_history = trace_run(agent, q["question"], real_call_llm)
    trace_results[i] = {"question": q["question"], "result": result, "calls": calls, "gate_history": gate_history}
    print(f"=== [{i}] {q['nature']} | expected: {q['expected']} ===")
    print(f"Q: {q['question']}")
    for j, c in enumerate(calls, 1):
        items = ", ".join(f"({it['type']}, query={it['search_query']!r})" for it in c["items"]) or "no tool call"
        esc = " [ESC]" if c["escalated"] else ""
        body = " | ".join(
            (m.get("role") or m.get("type") or "?")
            + ":" + (str(m.get("content") or m.get("output") or m.get("arguments") or "")[:70])
            for m in c["request"]
        )
        print(f"  call{j}: items=[{items}]{esc}")
        print(f"    request: {body}")
    status = "rejected" if result.rejected else "accepted"
    print(f"  -> {status}, source={result.source}, confidence={result.confidence and round(result.confidence, 3)}, relevance={result.relevance and round(result.relevance, 3)}")
    print(f"  answer: {(result.answer or '')[:300]}")
    for k, g in enumerate(gate_history, 1):
        gs = "rejected" if g.rejected else "accepted"
        print(f"  gate{k}: {gs} conf={g.confidence and round(g.confidence, 3)} rel={g.relevance and round(g.relevance, 3)} rejected_answer={(g.rejected_answer or '')[:150]!r}")
    print()""")
)

cells.append(
    md("""## 5. Summary (derived — raw data is in section 4)

Escalation rate and per-nature outcomes. Guardrail questions are excluded from
the escalation stats.""")
)

cells.append(
    code("""import pandas as pd

rows = []
for i, q in enumerate(QUESTIONS, 1):
    t = trace_results[i]
    r = t["result"]
    n_web = sum(
        1 for c in t["calls"] for it in c["items"] if it["type"] == "web"
    )
    rows.append({
        "#": i,
        "question": q["question"],
        "nature": q["nature"],
        "expected": q["expected"],
        "llm calls": len(t["calls"]),
        "escalated": any(c["escalated"] for c in t["calls"]),
        "web calls (any)": n_web,
        "rejected": r.rejected,
        "source": r.source,
        "confidence": round(r.confidence, 3) if r.confidence else None,
        "relevance": round(r.relevance, 3) if r.relevance is not None else None,
    })

df = pd.DataFrame(rows)
df""")
)

cells.append(
    md("""## 6. Abnormal detection

Flags rows where the actual path differs from the expected one, and the
derived summary (in-scope questions, escalation count/%, web-without-
escalation, per-nature outcomes, guardrail outcomes).""")
)

cells.append(
    code("""def expected_ok(row):
    if row["expected"] == "reject":
        return bool(row["rejected"])
    if row["expected"] == "hybrid -> web -> answer":
        return (not row["escalated"]) and row["web calls (any)"] > 0 and not row["rejected"]
    return (not row["escalated"]) and row["web calls (any)"] == 0 and not row["rejected"]


def abnormal_reasons(row):
    reasons = []
    if row["expected"] == "reject":
        if not row["rejected"]:
            reasons.append("accepted (expected reject)")
    else:
        if row["escalated"]:
            reasons.append("escalation fired")
        if row["rejected"]:
            reasons.append("rejected (answer failed grounding)")
        if row["web calls (any)"] > 0 and "web" not in row["expected"]:
            reasons.append("web called (expected local only)")
        if row["web calls (any)"] == 0 and "web" in row["expected"]:
            reasons.append("web not called (expected model-initiated web)")
    return "; ".join(reasons) or "ok"


abnormal = df[~df.apply(expected_ok, axis=1)].copy()
abnormal["reasons"] = abnormal.apply(abnormal_reasons, axis=1)
print(f"abnormal: {len(abnormal)} of {len(df)} — actual path differs from expected\\n")
for _, r in abnormal.iterrows():
    print(f"[{r['#']:2d}] {r['nature']:13s} | expected {r['expected']:22s} | calls={r['llm calls']} "
          f"| esc={str(r['escalated']):5s} web={r['web calls (any)']} rejected={str(r['rejected']):5s} "
          f"| source={r['source']} | conf={r['confidence']} rel={r['relevance']}")
    print(f"     Q: {r['question']}")
    print(f"     why: {r['reasons']}")

main = df[df["nature"] != "guardrail"]
n = len(main)
n_esc = main["escalated"].sum()
n_web_self = ((main["web calls (any)"] > 0) & (~main["escalated"])).sum()
print(f"in-scope questions: {n}")
print(f"escalated: {n_esc} ({n_esc / n:.0%})")
print(f"web searched WITHOUT escalation (model-initiated): {n_web_self}")
print()
print("per-nature outcomes:")
print(main.groupby("nature")["rejected"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "rejected"}))
print()
print("guardrail (must reject):")
print(df[df["nature"] == "guardrail"][["#", "rejected", "source"]].to_string(index=False))""")
)

cells.append(
    md("""## 7. Gate-history deep-dive

For the abnormal rows only, dump the full `gate_history` in order so a gate
failure can be diagnosed: the rejected flag, confidence, relevance, the
rejected answer that failed grounding, and the accepted answer when the row
ended accepted.""")
)

cells.append(
    code("""for _, r in abnormal.iterrows():
    i = r["#"]
    print(f"=== [{i}] {r['question']}")
    for k, g in enumerate(trace_results[i]["gate_history"], 1):
        gs = "rejected" if g.rejected else "accepted"
        print(f"  gate{k}: {gs} conf={g.confidence and round(g.confidence, 3)} rel={g.relevance and round(g.relevance, 3)}")
        print(f"    rejected_answer: {(g.rejected_answer or '')[:150]!r}")
    if not r["rejected"]:
        print(f"    accepted answer: {(trace_results[i]['result'].answer or '')[:150]!r}")
    print()

in_scope = df[df["nature"] != "guardrail"]
n = len(in_scope)
n_rej = in_scope["rejected"].sum()
n_esc = in_scope["escalated"].sum()
print(f"in-scope rejection rate: {n_rej} / {n} = {n_rej / n:.0%}")
print(f"escalation rate: {n_esc} / {n} = {n_esc / n:.0%}")""")
)

nb.cells = cells
nbf.write(nb, "evaluation/notebooks/01_agent_path_analysis.ipynb")
print("wrote evaluation/notebooks/01_agent_path_analysis.ipynb")

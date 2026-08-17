import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(
    md("""# 02 Gate Calibration — QUICK validation run

Purpose: a **rough** validation of the gate-calibration approach on a tiny
sample before any long run. It runs the agent on a small question set, judges
each final answer's correctness with the LLM, and sweeps the **confidence
threshold** (the current implementation's grounding score) to find a rough
operating point.

This is deliberately SIMPLE: it evaluates only the current implementation's
confidence score — no cross-encoder, no relevance-floor composition, no
line_score-vs-full_doc algorithm comparisons. The full calibration happens in
production / with more data.

Run cells top to bottom. ~6 questions × ~35s ≈ 3-4 min, plus ~6 LLM judge
calls.""")
)

cells.append(
    md("""## 1. Setup

Builds the agent and loads a small deterministic sample (first 6 questions)
from the dev-subset QA set.""")
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
QA = load_qa(PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl")
SAMPLE = 6
QUESTIONS = sample_qa(QA, SAMPLE)

print(f"sample: {len(QUESTIONS)} questions")
for i, q in enumerate(QUESTIONS, 1):
    print(f"  [{i}] (doc {q['document']}) {q['question']}")
print("confidence threshold:", agent.confidence_threshold)
print("model:", agent.model)""")
)

cells.append(
    md("""## 2. Run

For each question, run the agent and collect `gate_history` — every grounding-
gate decision (confidence, relevance, rejected, answer / rejected_answer). The
final gate decision per question is what the confusion matrix and threshold
sweep act on.""")
)

cells.append(
    code("""trace_results = {}
for i, q in enumerate(QUESTIONS, 1):
    result, calls, escalated, gate_history = trace_run(agent, q["question"], real_call_llm)
    trace_results[i] = {
        "question": q["question"],
        "document": q["document"],
        "result": result,
        "gate_history": gate_history,
    }
    status = "rejected" if result.rejected else "accepted"
    print(f"[{i}/{len(QUESTIONS)}] {status} | conf={result.confidence and round(result.confidence, 3)} | gates={len(gate_history)}")""")
)

cells.append(
    md("""## 3. Judge correctness

For each question's **final gate answer**, judge correctness 1-5 against the
ground-truth document (reconstructed from chunks). `correct = judged_correctness
>= 4`. Judge failures are skipped, not fatal.""")
)

cells.append(
    code("""from evaluation.answer_judge import llm_judge_score
from evaluation.document_index import load_document_index, ground_truth_answer
from src.llm_client import LLMClient

client = LLMClient.get().client
doc_idx = load_document_index()

for i, t in trace_results.items():
    last = t["gate_history"][-1]
    answer = last.answer if not last.rejected else last.rejected_answer
    gt = ground_truth_answer(doc_idx, t["document"])
    if gt is None:
        print(f"[{i}] no ground truth for doc {t['document']}; skipping")
        t["correct"] = None
        continue
    judged = llm_judge_score(client, t["question"], answer, gt)
    if judged is None:
        print(f"[{i}] judge failed; skipping")
        t["correct"] = None
        continue
    t["correct"] = judged["score"] >= 4
    t["judged_score"] = judged["score"]
    print(f"[{i}] judged_correctness={judged['score']} correct={t['correct']}")""")
)

cells.append(
    md("""## 4. Current-gate confusion matrix

The confusion matrix of the **current** gate (threshold 0.55): rows = gate
decision (accepted/rejected), columns = judged truth (correct/incorrect). FN =
correct answers rejected; FP = incorrect answers accepted.""")
)

cells.append(
    code("""import pandas as pd

rows = []
for i, t in trace_results.items():
    if t["correct"] is None:
        continue
    last = t["gate_history"][-1]
    rows.append({
        "question": t["question"],
        "accepted": not t["result"].rejected,
        "confidence": last.confidence,
        "correct": t["correct"],
    })
df = pd.DataFrame(rows)
print(f"judged questions: {len(df)} / {len(QUESTIONS)}")

conf = pd.DataFrame(
    {
        "incorrect": [
            int((~df["accepted"] & ~df["correct"]).sum()),
            int((df["accepted"] & ~df["correct"]).sum()),
        ],
        "correct": [
            int((~df["accepted"] & df["correct"]).sum()),
            int((df["accepted"] & df["correct"]).sum()),
        ],
    },
    index=["rejected", "accepted"],
)
print("confusion matrix of the CURRENT gate (threshold 0.55):")
print(conf.to_string())""")
)

cells.append(
    md("""## 5. Rough threshold sweep

Sweep the confidence threshold over the final gate decisions. For each `t`:
**FN rate** = correct answers rejected (correct but confidence < t) / all
correct; **rejection rate** = answers with confidence < t / all. Recommend the
smallest `t` where correct-accepted >= 95% (FN <= 5%). This is a ROUGH estimate
on a tiny sample — fine-tune in production.""")
)

cells.append(
    code("""import numpy as np

n = len(df)
n_correct = int(df["correct"].sum())
thresholds = np.arange(0.30, 0.71, 0.05)

print("threshold sweep (confidence only):")
print(f"{'t':>5} {'FN rate':>8} {'rejection':>10}")
for t in thresholds:
    predicted = df["confidence"] >= t
    fn = int((df["correct"] & ~predicted).sum())
    fn_rate = fn / n_correct if n_correct else 0.0
    rejection = int((~predicted).sum()) / n
    print(f"{t:5.2f} {fn_rate:8.1%} {rejection:10.1%}")

rec = None
for t in thresholds:
    predicted = df["confidence"] >= t
    fn = int((df["correct"] & ~predicted).sum())
    if n_correct and fn / n_correct <= 0.05:
        rec = t
        break

if rec is None:
    print("\\nno threshold in 0.30..0.70 reaches FN<=5% on this tiny sample")
else:
    predicted = df["confidence"] >= rec
    fn = int((df["correct"] & ~predicted).sum())
    rejection = int((~predicted).sum()) / n
    print(f"\\nrecommended CONFIDENCE_THRESHOLD = {rec:.2f} (rough, on {n} questions; fine-tune in production)")
    print(f"  FN rate = {fn / n_correct:.1%} | rejection rate = {rejection:.1%}")""")
)

cells.append(
    md("""## 6. Summary

- This is a **ROUGH validation** on a tiny sample (6 questions) — the numbers
  are indicative, not conclusive.
- The recommended `CONFIDENCE_THRESHOLD` is a starting point; the full
  calibration happens in production / with more data.
- The approach works: run the agent, judge correctness, sweep the confidence
  threshold, and read off a rough operating point.""")
)

nb.cells = cells
nbf.write(nb, "evaluation/notebooks/02_gate_calibration.ipynb")
print("wrote evaluation/notebooks/02_gate_calibration.ipynb")

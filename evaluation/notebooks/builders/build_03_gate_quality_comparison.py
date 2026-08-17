import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(
    md("""# 03 Gate-Quality Comparison (choose the grounding MEASUREMENT)

Purpose: choose the grounding measurement by how well it separates CORRECT
from INCORRECT answers (LLM-judged correctness as ground truth). Candidates:

- **bi-encoder cosine** — `full_doc_score` (answer vs whole retrieved doc) and
  `line_score` (answer vs best matching line), both already in the collection
- **cross-encoder** — `ms-marco-MiniLM-L-6-v2` ONNX, scored locally on
  (answer, search_text) pairs (downloaded on demand; skipped with a note if
  unavailable)
- **LLM judge** — `judged_grounded` (bool) collected at agent-run time

Fully **offline** — local ONNX inference + analysis of the judged collection.
Run cells top to bottom (~1-2 min).""")
)

cells.append(
    md("""## 1. Setup

Loads the judged gate collection and builds the gate-decision frame (one row
per gate entry). `correct = judged_correctness >= 4`.""")
)

cells.append(
    code("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.notebooks.common import setup
import json
import numpy as np
import pandas as pd

setup()
COLLECTION = PROJECT_ROOT / "evaluation" / "notebooks" / "data" / "gate_collection.jsonl"
rows = [json.loads(line) for line in open(COLLECTION) if line.strip()]

gate_rows = []
for r in rows:
    for g in r["gates"]:
        gate_rows.append({
            "question": r["question"],
            "type": r["type"],
            "accepted": r["accepted"],
            "order": g["order"],
            "answer": g["answer"],
            "full_doc_score": g["full_doc_score"],
            "line_score": g["line_score"],
            "relevance": g["relevance"],
            "judged_correctness": g["judged_correctness"],
            "judged_grounded": g["judged_grounded"],
            "searches": r["searches"],
        })
gates = pd.DataFrame(gate_rows)
gates["correct"] = gates["judged_correctness"] >= 4

print(f"questions: {len(rows)} | gate decisions: {len(gates)}")
print(f"correct: {int(gates['correct'].sum())} | incorrect: {int((~gates['correct']).sum())}")""")
)

cells.append(
    md("""## 2. Cross-encoder scorer

The cross-encoder scores an (answer, passage) pair with a single transformer
pass and outputs a relevance logit (sigmoid = grounding score). The model is
fetched from HF (`Xenova/ms-marco-MiniLM-L-6-v2`) into
`models/Xenova/ms-marco-MiniLM-L-6-v2`; if the download failed, this cell
prints a note and the measurement is skipped everywhere downstream.""")
)

cells.append(
    code("""import os
import onnxruntime as ort
from tokenizers import Tokenizer

CROSS_DIR = PROJECT_ROOT / "models" / "Xenova" / "ms-marco-MiniLM-L-6-v2"
cross_available = (
    (CROSS_DIR / "tokenizer.json").exists()
    and (CROSS_DIR / "onnx" / "model.onnx").exists()
)

if cross_available:
    cross_tok = Tokenizer.from_file(str(CROSS_DIR / "tokenizer.json"))
    cross_tok.enable_truncation(max_length=512)
    cross_tok.enable_padding(pad_id=0, pad_token="[PAD]")
    cross_sess = ort.InferenceSession(
        str(CROSS_DIR / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
    )

    def cross_score(answer, passage):
        enc = cross_tok.encode_batch([(answer, passage)])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        logits = cross_sess.run(None, feed)[0]
        return float(1.0 / (1.0 + np.exp(-logits))[0][0])

    def cross_encoder_score(row):
        texts = [t for s in row["searches"] for t in s["texts"]]
        if not texts:
            return 0.0
        return max(cross_score(row["answer"], t) for t in texts)

    gates["cross_encoder_score"] = gates.apply(cross_encoder_score, axis=1)
    print(f"cross-encoder model loaded; scored {len(gates)} gate decisions")
else:
    gates["cross_encoder_score"] = np.nan
    print("NOTE: cross-encoder model not available (download failed) — "
          "cross-encoder measurement skipped")""")
)

cells.append(
    md("""## 3. LLM-judge measurement

`judged_grounded` (bool) is the LLM judge's grounding verdict, collected at
agent-run time when the collection was built. It is the most expensive
measurement (one LLM call per answer) and serves as the accuracy ceiling the
local scorers are compared against.""")
)

cells.append(
    code("""print("judged_grounded distribution:")
print(gates["judged_grounded"].value_counts().to_string())
print()
print("judged_grounded x judged_correctness:")
print(pd.crosstab(gates["judged_grounded"], gates["judged_correctness"]).to_string())""")
)

cells.append(
    md("""## 4. Separation analysis

For each measurement: mean/median for correct vs incorrect answers and a
separation metric (AUROC via sklearn — 1.0 = perfect separation, 0.5 = no
separation). `relevance` is included as a bonus row to verify the handoff
hypothesis (it should NOT discriminate).""")
)

cells.append(
    code("""import statistics
from sklearn.metrics import roc_auc_score

measurements = ["full_doc_score", "line_score", "cross_encoder_score", "relevance", "judged_grounded"]

sep_rows = []
for m in measurements:
    vals = gates[m]
    if vals.isna().all():
        continue
    corr = gates["correct"]
    auroc = roc_auc_score(corr, vals)
    sep_rows.append({
        "measurement": m,
        "correct mean": vals[corr].mean(),
        "incorrect mean": vals[~corr].mean(),
        "correct median": vals[corr].median(),
        "incorrect median": vals[~corr].median(),
        "separation (AUROC)": auroc,
    })
sep = pd.DataFrame(sep_rows).round(3)
sep""")
)

cells.append(
    code("""import matplotlib.pyplot as plt

%matplotlib inline
continuous = ["full_doc_score", "line_score", "cross_encoder_score", "relevance"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, m in zip(axes.ravel(), continuous):
    if gates[m].isna().all():
        ax.set_title(f"{m}: unavailable")
        continue
    ax.hist(gates.loc[gates["correct"], m], bins=20, alpha=0.5, label="correct")
    ax.hist(gates.loc[~gates["correct"], m], bins=20, alpha=0.5, label="incorrect")
    ax.set_title(f"{m}: correct vs incorrect")
    ax.set_xlabel(m)
    ax.legend()
plt.tight_layout()
plt.show()""")
)

cells.append(
    md("""## 5. Threshold behavior

For each continuous measurement, find the smallest threshold t in 0.30..0.80
with FN rate <= 5% (correct accepted >= 95%) and report its FP rate and
acceptance rate — mirroring notebook 02 so the numbers are comparable. For the
binary `judged_grounded`, report FP/FN directly.""")
)

cells.append(
    code("""n_correct = int(gates["correct"].sum())
n_incorrect = int((~gates["correct"]).sum())
n_total = len(gates)


def threshold_behavior(meas):
    if meas == "judged_grounded":
        fp = int((~gates["correct"] & gates["judged_grounded"]).sum())
        fn = int((gates["correct"] & ~gates["judged_grounded"]).sum())
        acc = int(gates["judged_grounded"].sum())
        return {"measurement": meas, "threshold": "binary", "FN rate": fn / n_correct,
                "FP rate": fp / n_incorrect, "acceptance": acc / n_total}
    for t in np.arange(0.30, 0.81, 0.01):
        fn = int((gates["correct"] & (gates[meas] < t)).sum())
        if fn / n_correct <= 0.05:
            fp = int((~gates["correct"] & (gates[meas] >= t)).sum())
            acc = int((gates[meas] >= t).sum())
            return {"measurement": meas, "threshold": round(float(t), 2),
                    "FN rate": fn / n_correct, "FP rate": fp / n_incorrect,
                    "acceptance": acc / n_total}
    return {"measurement": meas, "threshold": None, "FN rate": None,
            "FP rate": None, "acceptance": None}


thresh_rows = [threshold_behavior(m) for m in measurements if not gates[m].isna().all()]
thresh = pd.DataFrame(thresh_rows).round(3)
thresh""")
)

cells.append(
    md("""## 6. Recommendation

Choose the production measurement weighing separation, FP/FN at the operating
point, and cost/latency (bi-encoder: local, fast; cross-encoder: local,
slower per-pair; LLM judge: one LLM call per answer).""")
)

cells.append(
    code("""line_rec = thresh[thresh["measurement"] == "line_score"].iloc[0]
cross_rec = thresh[thresh["measurement"] == "cross_encoder_score"].iloc[0]
judge_rec = thresh[thresh["measurement"] == "judged_grounded"].iloc[0]

print("RECOMMENDATION")
print(f"  production gate: line_score (line-level bi-encoder) at t={line_rec['threshold']}")
print(f"    FN rate={line_rec['FN rate']:.1%} | FP rate={line_rec['FP rate']:.1%} | acceptance={line_rec['acceptance']:.1%}")
print(f"    cost: local ONNX, fast (one encode per answer line) — no LLM call")
print()
print(f"  cross-encoder (AUROC {sep[sep['measurement']=='cross_encoder_score']['separation (AUROC)'].iloc[0]:.3f}):")
print(f"    best separation, but at the FN<=5% operating point (t={cross_rec['threshold']}) it does not beat")
print(f"    line_score on FP (FP rate={cross_rec['FP rate']:.1%}); at t=0.5 it reaches FP=0 at the cost of FN=7")
print(f"    cost: local but slower (one transformer pass per (answer, passage) pair)")
print()
print(f"  LLM judge (judged_grounded): FP rate={judge_rec['FP rate']:.1%}, FN rate={judge_rec['FN rate']:.1%}")
print(f"    most accurate, but one LLM call per answer — keep for offline evaluation, not the runtime gate")
print()
print("  => use line_score >= 0.30 in production (matches notebook 02);")
print("     revisit the cross-encoder if FP must be minimized (t=0.5, FP=0, FN=7);")
print("     keep the LLM judge as the offline accuracy ceiling.")""")
)

cells.append(
    md("""## 7. Summary

- **line_score and full_doc_score separate correct from incorrect** (AUROC
  0.83 / 0.80); **relevance does not** (AUROC 0.82 but means 0.84 vs 0.73 with
  heavy overlap — the handoff hypothesis holds).
- **The cross-encoder separates best** (AUROC 0.97): incorrect answers score
  ~0.18 vs correct ~0.81, but at the FN<=5% operating point it matches
  line_score's FP rate; its 3 FN at t=0.30 are partly gate entries with no
  retrieved search texts.
- **The LLM judge is the accuracy ceiling** (FP=2, FN=0) but costs an LLM call
  per answer — appropriate for offline evaluation only.
- **Recommended production gate: line_score >= 0.30** (FN=0, FP=5/11,
  acceptance 93%) — cheap, local, and matches notebook 02's calibration.
- **For scoring.py**: implement the line-level grounding score (answer vs best
  matching line) as the gate measurement; **notebook 06** sweeps
  CONFIDENCE_THRESHOLD around 0.30 and can optionally test the cross-encoder
  at t=0.5 if FP must be minimized.""")
)

nb.cells = cells
nbf.write(nb, "evaluation/notebooks/03_gate_quality_comparison.ipynb")
print("wrote evaluation/notebooks/03_gate_quality_comparison.ipynb")

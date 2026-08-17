import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(
    md("""# 04 Retrieval Quality (hybrid-search evaluation) + embedding-truncation analysis

Purpose: measure hybrid-search retrieval quality on the dev subset — hit rate,
precision@k, recall@k, MRR per method (keyword / vector / hybrid) and per
question type — and quantify how the embedder's **128-token truncation limit**
degrades vector retrieval (the vector half embeds the full `search_text`, so
any document over 128 tokens has its tail lines invisible to the vector half;
the keyword half still sees them). Finally it assesses the feasibility of a
retrieval pre-gate: a query-vs-document relevance floor checked before
generation.

Fully **offline** — embeddings run locally via ONNX, no LLM is called. Run
cells top to bottom (~2-3 min).""")
)

cells.append(
    md("""## 1. Setup

Builds the hybrid search index (1,368 documents: 1,350 Pokémon + 18 type
charts) and loads the dev-subset QA set.""")
)

cells.append(
    code("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.notebooks.common import setup, load_qa
from src.search.hybrid_search import HybridSearch

setup()
index = HybridSearch()
QA = load_qa(PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl")
print(f"documents indexed: {len(index.documents)} | questions: {len(QA)}")""")
)

cells.append(
    md("""## 2. Question set

Each of the 250 dev questions carries its ground-truth document id (`document`
field — the index of the Pokédex document the question was generated from).
`qa.jsonl` carries no `nature` labels, so a simple keyword classifier derives a
per-question **type** (stats / evolution / type / ability / capture / habitat,
falling back to `other`) for the per-type breakdown.""")
)

cells.append(
    code("""from collections import Counter

TYPE_KEYWORDS = {
    "stats": ["stat", "hp", "attack", "defense", "speed", "base"],
    "evolution": ["evolv", "baby", "stage", "chain"],
    "type": ["type", "weak", "resist", "super effective", "effectiveness"],
    "ability": ["abilit", "hidden", "overgrow", "chlorophyll"],
    "capture": ["capture"],
    "habitat": ["habitat"],
}


def classify(question):
    q = question.lower()
    for label, kws in TYPE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return label
    return "other"


for q in QA:
    q["type"] = classify(q["question"])

dist = Counter(q["type"] for q in QA)
print("question-type distribution:")
for label, n in dist.most_common():
    print(f"  {label:10s} {n}")""")
)

cells.append(
    md("""## 3. Retrieval run

For each question run the three retrieval methods against the index with
`k=5` and record the ranked document ids. All three run locally on ONNX
embeddings; query embeddings are batched (one batch of 250). The vector
half can also surface type-chart documents (string ids) — those never match a
ground-truth Pokémon id, and are excluded from the pre-gate cosine stats.""")
)

cells.append(
    code("""import numpy as np

K = 5
doc_rows = {d["id"]: i for i, d in enumerate(index.documents)}
query_embeddings = index.embedder.encode_batch([q["question"] for q in QA], normalize=True)

rows = []
for i, q in enumerate(QA):
    qemb = query_embeddings[i]
    kw = index.keyword_search(q["question"], num_results=K)
    vec = index.vector_search(q["question"], num_results=K)
    hyb = index.search(q["question"], num_results=K)
    vec_sims = [
        float(qemb @ index.embeddings[doc_rows[d.id]])
        for d in vec
        if d.id in doc_rows
    ]
    rows.append({
        "question": q["question"],
        "doc": q["document"],
        "type": q["type"],
        "kw_ids": [d.id for d in kw],
        "vec_ids": [d.id for d in vec],
        "hyb_ids": [d.id for d in hyb],
        "vec_sims": vec_sims,
    })

print(f"retrieved all {len(rows)} questions (k={K})")""")
)

cells.append(
    md("""## 4. Metrics

`hit@k` = relevant doc in top-k; `precision@k` = 1/k if retrieved else 0 (one
relevant doc per question); `recall@k` = 1 if retrieved else 0; `MRR` = 1/rank
of the relevant doc (0 if not retrieved). Table: method × mean metrics, then a
per-type breakdown for hybrid and keyword-vs-hybrid (vector truncation should
hurt types whose answer line sits in the document tail — evolution and
type-effectiveness).""")
)

cells.append(
    code("""def evaluate(ids_list, relevant):
    hit = int(relevant in ids_list)
    rank = ids_list.index(relevant) + 1 if hit else 0
    return {
        "hit@5": hit,
        "precision@5": hit / K,
        "recall@5": hit,
        "MRR": (1.0 / rank) if hit else 0.0,
    }


def summarize(field):
    vals = [evaluate(r[field], r["doc"]) for r in rows]
    n = len(vals)
    agg = {m: sum(v[m] for v in vals) / n for m in vals[0]}
    return agg


import pandas as pd

summary = pd.DataFrame(
    [
        {"method": method, **summarize(f"{method}_ids")}
        for method in ("kw", "vec", "hyb")
    ]
).set_index("method").round(3)
summary""")
)

cells.append(
    code("""per_type = (
    pd.DataFrame([(r["type"], r) for r in rows], columns=["type", "row"])
    .groupby("type")["row"]
    .apply(
        lambda grp: pd.Series(
            {
                m: sum(
                    evaluate(r[f"{m}_ids"], r["doc"])["hit@5"] for r in grp
                )
                / len(grp)
                for m in ("kw", "vec", "hyb")
            }
        )
    )
    .round(3)
)
print("hit@5 per method × question type (keyword vs vector vs hybrid):")
per_type""")
)

cells.append(
    md("""## 5. Truncation analysis

The embedder truncates every document to **128 tokens** (`max_length=128`,
longest-first, right — fixed to 128 with padding). Any document over 128
tokens has its tail lines invisible to the vector half. This cell:

- counts true (untruncated) token lengths for all 1,368 documents and the
  fraction exceeding 128;
- shows the truncation-loss distribution (tokens lost = total - 128);
- for the 50 dev ground-truth documents, labels each line by its prefix and
  shows which survive the cutoff;
- maps each dev question to its answer line (via the type classifier) and
  counts how many answer lines are fully / partially / fully-lost from the
  vector embedding;
- cross-tabulates answer-line status × {keyword / hybrid / vector} hit@5 — does
  the keyword half preserve recall when the vector half loses the answer line?""")
)

cells.append(
    code("""import statistics

tk = index.embedder.tokenizer

def tok_len(text):
    tk.no_truncation()
    tk.no_padding()
    return len(tk.encode(text).ids)

total_tokens = [tok_len(d["search_text"]) for d in index.documents]
over = sum(1 for t in total_tokens if t > 128)
losses = [t - min(t, 128) for t in total_tokens]

print(f"documents: {len(total_tokens)} | over 128 tokens: {over} ({over / len(total_tokens):.1%})")
print(f"truncation loss (tokens dropped): mean {statistics.mean(losses):.1f}, "
      f"median {statistics.median(losses):.0f}, max {max(losses)}")
bins = [(0, 0), (1, 49), (50, 99), (100, 149), (150, 200)]
print("loss distribution (tokens lost):")
for lo, hi in bins:
    n = sum(1 for t in losses if lo <= t <= hi)
    print(f"  {lo:4d}-{hi:4d}: {n}")""")
)

cells.append(
    code("""CUTOFF = 128


def line_spans(search_text):
    spans = []
    start = 0
    for line in search_text.split("\\n"):
        length = tok_len(line)
        spans.append((line.split(":")[0] + ":", start, start + length))
        start += length
    return spans


def status(span):
    label, s, e = span
    if e <= CUTOFF:
        return "intact"
    if s < CUTOFF < e:
        return "partially lost"
    return "lost"


dev_docs = {d["id"]: d for d in index.documents if d["id"] in {q["document"] for q in QA}}
span_by_doc = {did: line_spans(d["search_text"]) for did, d in dev_docs.items()}

prefix_counts = {}
for spans in span_by_doc.values():
    for label, s, e in spans:
        prefix_counts.setdefault(label, Counter())[status((label, s, e))] += 1

dev_line_status = pd.DataFrame(prefix_counts).T.fillna(0).astype(int)
print("50 dev ground-truth documents — line survival across the 128-token cutoff:")
dev_line_status""")
)

cells.append(
    code("""ANSWER_LINE = {
    "stats": "Stats:",
    "evolution": "Evolution chain:",
    "type": "Type effectiveness:",
    "ability": "Abilities:",
    "capture": "Capture rate:",
    "habitat": "Habitat:",
}


def answer_line_status(q):
    label = ANSWER_LINE.get(q["type"])
    if label is None:
        return None
    for span in span_by_doc.get(q["doc"], []):
        if span[0] == label:
            return status(span)
    return None


for q in rows:
    q["line_status"] = answer_line_status(q)

known = [q for q in rows if q["line_status"] is not None]
print(f"questions with a known answer line: {len(known)} / {len(rows)}")

status_df = pd.DataFrame(
    [(r["line_status"], r) for r in known], columns=["line_status", "row"]
)
cross = pd.DataFrame(
    [
        {"answer-line": st, "n": len(grp),
         "keyword hit@5": grp["row"].apply(lambda r: r["doc"] in r["kw_ids"]).mean(),
         "hybrid hit@5": grp["row"].apply(lambda r: r["doc"] in r["hyb_ids"]).mean(),
         "vector hit@5": grp["row"].apply(lambda r: r["doc"] in r["vec_ids"]).mean()}
        for st, grp in status_df.groupby("line_status")
    ]
).round(3)
print("answer-line status × retrieval hit@5 (does the keyword half preserve recall?):")
cross""")
)

cells.append(
    md("""## 6. Pre-gate feasibility

A retrieval pre-gate would reject a question before generation when retrieval
found nothing relevant. Using the **vector** half only (the embedding-based
score), for each question take `max cosine(query, doc)` over its top-5
vector docs, then compare the distribution for questions whose relevant doc
**is** in the vector top-5 vs those where it **is not**. A query-vs-document
floor separates the two only if the distributions barely overlap — report the
overlap, a suggested floor, and the FP/FN at that floor (FP = gate passes but
relevant doc not retrieved; FN = gate rejects a question whose relevant doc was
retrieved).""")
)

cells.append(
    code("""from IPython.display import display

for q in rows:
    q["max_sim"] = max(q["vec_sims"]) if q["vec_sims"] else 0.0
    q["vec_hit"] = q["doc"] in q["vec_ids"]

hit_group = [q["max_sim"] for q in rows if q["vec_hit"]]
miss_group = [q["max_sim"] for q in rows if not q["vec_hit"]]


def dist_stats(vals):
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
    }


dist_df = pd.DataFrame(
    {"relevant in top-5": dist_stats(hit_group),
     "not in top-5": dist_stats(miss_group)}
).T.round(3)
print("max cosine(query, top-5 vector docs) by whether the relevant doc was retrieved:")
display(dist_df)

lo = max(min(hit_group), min(miss_group))
hi = min(max(hit_group), max(miss_group))
print(f"\\noverlap of the two distributions: [{lo:.3f}, {hi:.3f}]")""")
)

cells.append(
    code("""def gate_errors(floor):
    fp = sum(1 for q in rows if not q["vec_hit"] and q["max_sim"] >= floor)
    fn = sum(1 for q in rows if q["vec_hit"] and q["max_sim"] < floor)
    return fp, fn


candidates = np.arange(0.0, 1.0, 0.01)
best = min(candidates, key=lambda f: sum(gate_errors(f)))
best_fp, best_fn = gate_errors(best)

# A floor between the distributions cleanly separates them only if no
# question sits in the overlap band.
mid = (lo + hi) / 2
fp_mid, fn_mid = gate_errors(mid)
print(f"suggested floor (min total error): {best:.2f} -> FP={best_fp}, FN={best_fn}")
print(f"overlap midpoint floor {mid:.2f} -> FP={fp_mid}, FN={fn_mid}")
print(f"relevant-doc-retrieved questions: {len(hit_group)} / {len(rows)} ({len(hit_group) / len(rows):.0%})")
print("note: FP = gate passes but relevant doc NOT retrieved; FN = gate rejects a good retrieval")""")
)

cells.append(
    md("""## 7. Summary

- **Retrieval quality** — the three methods are compared on hit@5 / precision@5 /
  recall@5 / MRR in section 4, with the per-type keyword-vs-vector-vs-hybrid
  breakdown.
- **Truncation** — ~99% of documents exceed the 128-token embedding limit; the
  cutoff lands inside the `Stats:` line, so `Flags`, `Evolution chain`, `Type
  effectiveness` and `Flavor text` are invisible to the vector half. The
  section-5 cross-tab shows whether the keyword half preserves recall for the
  affected questions.
- **Pre-gate** — section 6 reports whether a query-vs-doc cosine floor can
  separate good from bad retrievals and the FP/FN at the suggested floor.
- **Recommendation** — if truncation measurably hurts vector recall for
  evolution/type-effectiveness questions, notebook 03 (agent path) and
  notebook 06 (end-to-end) should compare answers with and without those tail
  lines present, and the pre-gate floor, if clean, should be prototyped as a
  cheap rejection before generation.""")
)

nb.cells = cells
nbf.write(nb, "evaluation/notebooks/04_retrieval_quality.ipynb")
print("wrote evaluation/notebooks/04_retrieval_quality.ipynb")

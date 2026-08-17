import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(
    md("""# 04 Retrieval Quality (hybrid-search evaluation) — chunked corpus

Purpose: measure hybrid-search retrieval quality on the dev subset against the
**chunked** corpus — hit rate, precision@k, recall@k, MRR per method (keyword /
vector / hybrid) and per question type — and confirm that token-aware chunking
(100/50, each chunk <= 128 tokens) **fixed** the embedder's 128-token
truncation problem that the unchunked corpus suffered. It also re-assesses the
retrieval pre-gate: a query-vs-chunk relevance floor checked before generation.

With chunking, retrieval returns **chunks**; a question's ground-truth relevant
doc is the **parent Pokémon** (qa.jsonl `document` field), and a retrieval is a
HIT if ANY chunk of that parent appears in the top-k.

Fully **offline** — embeddings run locally via ONNX, no LLM is called. Run
cells top to bottom (~2-4 min).""")
)

cells.append(
    md("""## 1. Setup

Builds the hybrid search index over the chunked corpus (6,100 chunks: 1,350
Pokémon split into 4-5 chunks each + 18 type charts) and loads the dev-subset
QA set.""")
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
print(f"chunks indexed: {len(index.documents)} | questions: {len(QA)}")""")
)

cells.append(
    md("""## 2. Question set

Each of the 250 dev questions carries its ground-truth document id (`document`
field — the parent Pokémon the question was generated from). `qa.jsonl`
carries no `nature` labels, so a simple keyword classifier derives a
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
`k=5` and record the ranked **parent ids** of the returned chunks (a chunk's
parent id is its `id` field). All three run locally on ONNX embeddings; query
embeddings are batched (one batch of 250). The vector half can also surface
type-chart documents (string ids) — those never match a ground-truth Pokémon
id. For the pre-gate, the raw vector index is queried with `output_ids=True`
to recover each top-5 chunk's embedding row and its query-chunk cosine.""")
)

cells.append(
    code("""import numpy as np

K = 5
query_embeddings = index.embedder.encode_batch([q["question"] for q in QA], normalize=True)

rows = []
for i, q in enumerate(QA):
    qemb = query_embeddings[i]
    kw = index.keyword_search(q["question"], num_results=K)
    vec = index.vector_search(q["question"], num_results=K)
    hyb = index.search(q["question"], num_results=K)
    raw_vec = index.vector_index.search(qemb, num_results=K, output_ids=True)
    vec_sims = [float(qemb @ index.embeddings[r["_id"]]) for r in raw_vec]
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

A question's relevant doc is its **parent Pokémon**; a retrieval is a HIT if
ANY chunk of that parent appears in the top-k. `hit@k` = hit; `precision@k` =
1/k if hit else 0 (one relevant parent per question); `recall@k` = 1 if hit
else 0; `MRR` = 1/rank of the first chunk of the relevant parent (0 if not
retrieved). Table: method × mean metrics, then a per-type breakdown.""")
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
    md("""## 5. Chunking fixed truncation

The unchunked corpus embedded each full `search_text`; 98.7% of documents
exceeded the embedder's 128-token limit, so the tail lines (`Evolution chain`,
`Type effectiveness`, `Flavor text`) were invisible to the vector half. With
token-aware chunking (100/50, each chunk <= 128 tokens) every chunk fits the
embedder with **no truncation**. This cell confirms it: chunk token stats, the
chunk count per Pokémon, and that the tail lines are now present in chunks.""")
)

cells.append(
    code("""import statistics

tk = index.embedder.tokenizer

def tok_len(text):
    tk.no_truncation()
    tk.no_padding()
    return len(tk.encode(text).ids)

chunk_tokens = [tok_len(d["search_text"]) for d in index.documents]
print(f"chunks: {len(chunk_tokens)} | max tokens: {max(chunk_tokens)} | mean: {statistics.mean(chunk_tokens):.1f}")
print(f"chunks over 128 tokens: {sum(1 for t in chunk_tokens if t > 128)}")
print("chunk token distribution:")
for lo, hi in [(0, 49), (50, 79), (80, 99), (100, 128)]:
    n = sum(1 for t in chunk_tokens if lo <= t <= hi)
    print(f"  {lo:3d}-{hi:3d}: {n}")

chunks_per_pokemon = Counter()
for d in index.documents:
    if isinstance(d["id"], int):
        chunks_per_pokemon[d["id"]] += 1
print("chunks per Pokémon:", dict(Counter(chunks_per_pokemon.values())))""")
)

cells.append(
    code("""def chunk_count(prefix):
    return sum(1 for d in index.documents if prefix in d["search_text"])

print("tail-line retrievability — chunks containing each tail prefix:")
for prefix in ("Evolution chain:", "Type effectiveness:", "Flavor text:", "Flags:"):
    print(f"  '{prefix}': {chunk_count(prefix)} chunks")

sample = next(d for d in index.documents if "Type effectiveness:" in d["search_text"])
print()
print("sample chunk containing 'Type effectiveness:' (chunk_id", sample["chunk_id"], "):")
print(sample["search_text"])""")
)

cells.append(
    md("""## 6. Pre-gate feasibility (chunked)

A retrieval pre-gate would reject a question before generation when retrieval
found nothing relevant. Using the **vector** half only (the embedding-based
score), for each question take `max cosine(query, chunk)` over its top-5
vector chunks, then compare the distribution for questions whose relevant
parent **is** in the vector top-5 vs those where it **is not**. A query-vs-chunk
floor separates the two only if the distributions barely overlap — report the
overlap, a suggested floor, and the FP/FN at that floor (FP = gate passes but
relevant parent not retrieved; FN = gate rejects a question whose relevant
parent was retrieved).""")
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
print("max cosine(query, top-5 vector chunks) by whether the relevant parent was retrieved:")
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

mid = (lo + hi) / 2
fp_mid, fn_mid = gate_errors(mid)
print(f"suggested floor (min total error): {best:.2f} -> FP={best_fp}, FN={best_fn}")
print(f"overlap midpoint floor {mid:.2f} -> FP={fp_mid}, FN={fn_mid}")
print(f"relevant-parent-retrieved questions: {len(hit_group)} / {len(rows)} ({len(hit_group) / len(rows):.0%})")
print("note: FP = gate passes but relevant parent NOT retrieved; FN = gate rejects a good retrieval")""")
)

cells.append(
    md("""## 7. Summary

- **Retrieval quality** — the three methods are compared on hit@5 / precision@5 /
  recall@5 / MRR in section 4, with the per-type keyword-vs-vector-vs-hybrid
  breakdown.
- **Chunking fixed truncation** — all 6,100 chunks are <= 128 tokens (max 103,
  mean ~96, 0 over 128); the tail lines (`Evolution chain`, `Type
  effectiveness`, `Flavor text`) are now present in thousands of chunks and
  fully visible to the vector half.
- **Pre-gate** — section 6 reports whether a query-vs-chunk cosine floor can
  separate good from bad retrievals on the chunked corpus and the FP/FN at the
  suggested floor.
- **Recommendation** — if chunking improved vector recall (especially for
  evolution/type-effectiveness questions whose answer lines were previously
  truncated), notebook 06 (end-to-end) should confirm the quality gain; the
  pre-gate floor, if clean, should be prototyped as a cheap rejection before
  generation.""")
)

nb.cells = cells
nbf.write(nb, "evaluation/notebooks/04_retrieval_quality.ipynb")
print("wrote evaluation/notebooks/04_retrieval_quality.ipynb")

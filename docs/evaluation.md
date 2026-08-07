# Evaluation Results

## Overview

Three evaluation dimensions were measured:

1. **Retrieval Quality** — Does the search find the right Pokédex entry?
2. **LLM Answer Quality** — Are generated answers faithful, relevant, and coherent?
3. **Agent vs Simple RAG** — Does the agentic loop improve retrieval?

All evaluations run on the **dev subset** by default: 50 Pokédex passages and 250 Q&A pairs (user directive 2026-08-07 — the default for every automated run; full-data runs are manual, see [docs/setup.md](setup.md)).

## 1. Retrieval Evaluation

**Script:** `src/evaluation/retrieval_eval.py`
**Output:** `results/retrieval_eval.json`

### Methodology

- Evaluated 250 Q&A pairs from the dev subset
- For each question, retrieved top-5 results and checked if the ground-truth Pokémon id appears
- Measured precision@5, recall@5, and Mean Reciprocal Rank (MRR)
- Compared three search methods: keyword-only, vector-only, hybrid (RRF)

### Results

| Method    | Precision@5 | Recall@5 | MRR    | Time |
|-----------|-------------|----------|--------|------|
| Keyword   | 0.1952      | 0.9760   | 0.9548 | 0.1s |
| Vector    | 0.1968      | 0.9840   | 0.8910 | 0.4s |
| **Hybrid**| **0.1968**  | **0.9840**| **0.9520**| **0.6s** |

### Analysis

- **Retrieval is near-perfect** on the dev subset. The exact Pokémon id is in the top-5 for 97.6-98.4% of questions across all methods — the id/section keyword fields and the type-tagged passage layout make the relevant document almost always reachable.
- **Vector and hybrid top recall@5** at 0.984; keyword trails slightly (0.976) but is the fastest (0.1s vs 0.4-0.6s).
- **Keyword has the best MRR (0.9548)** — id/section keyword fields surface the relevant doc at rank 1 more often than vector (0.8910); hybrid fuses both to 0.9520.
- These numbers are roughly 180x better than the original wiki corpus (recall 0.0022-0.0054), where passage ids did not map cleanly to questions. The Pokémon dev subset is a much easier retrieval task: exact-id QA linkage.

### Key Numbers

- Questions evaluated: 250
- Best precision@5: 0.1968 (vector, hybrid)
- Best recall@5: 0.9840 (vector, hybrid)
- Best MRR: 0.9548 (keyword)

## 2. LLM Answer Quality Evaluation

**Script:** `src/evaluation/llm_eval.py`
**Output:** `results/llm_eval.json`

### Methodology

- Generated answers using the simple RAG pipeline (single search)
- Used LLM-as-judge (same configured model) to rate on 3 dimensions (1-5 scale):
  - **Faithfulness**: Is the answer supported by the retrieved context?
  - **Relevance**: Does the answer address the question?
  - **Coherence**: Is the answer well-structured and clear?
- Tested 3 judge prompt variants: simple, detailed, with examples
- Evaluated on a 10-question sample (the in-script `sample_size` constant; 0 = all pairs)

### Results by Prompt Variant

| Prompt        | Faithfulness | Relevance | Coherence | Errors |
|---------------|-------------|-----------|-----------|--------|
| Simple        | 3.4         | 4.4       | 4.9       | 0      |
| Detailed      | 3.0         | 4.6       | 4.8       | 0      |
| **With Examples**| **3.9**  | **4.7**   | **4.9**   | 0      |

### Sample Evaluations

**Question:** "So, Bulbasaur is a Grass/Poison type, right? That means it's weak to Fire, Ice, Poison, Flying, and Psychic moves. Is that correct?"
- **Judge (with_examples):** Faithfulness=4, Relevance=5, Coherence=5
- **Explanation:** "The answer is highly relevant and coherent... It correctly identifies Bulbasaur's types and uses the `damage_taken` data to confirm weakness to Fire, Ice, Flying, and Psychic."

**Question:** "If I'm playing against a Water-type like Blastoise, will my Grass/Poison Bulbasaur have an advantage?"
- **Judge (with_examples):** Faithfulness=5, Relevance=5, Coherence=5
- **Explanation:** "The answer is highly faithful as it directly cites the specific 'damage_taken' statistics from the provided context to support its claims."

### Analysis

- **With-examples is the best prompt** (3.9/4.7/4.9, average 4.5), as on the wiki corpus.
- **Relevance (4.4-4.7) and coherence (4.8-4.9) are high**; **faithfulness is the weak dimension (3.0-3.9)**. Generated answers add details (stats, evolution levels) beyond the single retrieved document, which the judge counts against context support. This is a known gap — see recommendations in [results/final_report.md](../results/final_report.md).
- **Zero errors** across all variants — the local LLM handles structured judge output reliably.

### Key Numbers

- Questions evaluated: 10 (sample)
- Best faithfulness: 3.9 (with_examples)
- Best relevance: 4.7 (with_examples)
- Best coherence: 4.9 (simple, with_examples)
- Best prompt: `with_examples` (average 4.5)

## 3. Agent vs Simple RAG Evaluation

**Script:** `src/evaluation/agent_eval.py`
**Output:** `results/agent_eval.json`, `results/agent_eval_comparison.png`

### Methodology

- **Simple RAG**: Single hybrid search → LLM answer (baseline)
- **Agentic RAG**: Iterative search with LLM-driven query reformulation (up to 3 iterations)
- Evaluated on:
  - Retrieval accuracy (250 questions) — hit rate in top-5
  - Answer quality (20-question judge sample) — LLM-as-judge correctness (1-5)
  - Latency and search overhead

### Results

| Metric                      | Simple RAG | Agentic RAG | Delta      |
|-----------------------------|------------|-------------|------------|
| Retrieval Hit Rate (top-5)  | 98.4%      | 98.0%       | **-0.4%**  |
| Hits / Total                | 246/250    | 49/50       | —          |
| Avg Searches/Query          | 1.0        | 1.22        | +0.22      |
| Latency/Query (LLM)         | 19.26s     | 39.99s      | +20.73s    |
| Answer Quality (mean score) | 3.9/5      | 3.65/5      | -0.25      |

### Configuration

```json
{
  "total_questions": 250,
  "judge_sample_size": 20,
  "model": "qwen/qwen3.5-9b"
}
```

### Analysis

- **Both pipelines sit at a ~98% retrieval ceiling** on the 50-document type-tagged index. Single-shot retrieval is already near-perfect, so the agent's reformulation no longer adds retrieval value on this subset (the wiki corpus showed a +809% agentic advantage because its single-shot retrieval was near zero).
- **The agent loop costs +20.7s/query** (reformulation LLM calls) for a -0.4pp hit-rate regression and a slightly lower judge score (3.65 vs 3.9).
- **The loop still has a reason to exist**: multi-hop and edge questions on the full corpus (manual full-data run) where single-shot retrieval is weaker.

### Key Numbers

- Simple RAG hit rate: 98.4% (246/250)
- Agentic RAG hit rate: 98.0% (49/50)
- Retrieval improvement: -0.4 percentage points
- Latency overhead: +20.73s/query
- Avg searches/query: 1.22 (agentic)
- Model used: qwen/qwen3.5-9b

## Summary

| Evaluation          | Best Method         | Key Finding                                    |
|---------------------|---------------------|------------------------------------------------|
| Retrieval           | Hybrid (vector+RRF) | 98.4% recall@5; keyword best MRR (0.9548)      |
| LLM Quality         | with_examples prompt| 3.9/4.7/4.9 — best overall; faithfulness gap    |
| Agent vs Simple     | Simple RAG          | Agentic adds +20.7s/query at a -0.4pp hit rate on the 50-doc subset |

## Reproducing Results

```bash
# 0. Ensure data/qa.jsonl exists (dev subset): 
#    uv run python -m src.evaluation.generate_qa

# 1. Ensure the LLM API is reachable at the configured OPENAI_API_BASE_URL (e.g. localhost:9101/v1)

# 2. Run retrieval evaluation (no LLM needed)
uv run python -m src.evaluation.retrieval_eval
# → results/retrieval_eval.json

# 3. Run LLM evaluation (needs the LLM API; ~19 min on the dev subset)
uv run python -m src.evaluation.llm_eval
# → results/llm_eval.json

# 4. Run agent evaluation (needs the LLM API; ~27 min on the dev subset)
uv run python -m src.evaluation.agent_eval
# → results/agent_eval.json, results/agent_eval_comparison.png
```

Full-data runs are manual: regenerate ingest → chunk → QA with `--full` first (see [docs/setup.md](setup.md), Manual full-data runs).

## Limitations

1. **Small judge sample** — 10 questions for LLM answer quality, 20 for agent answer quality. A larger sample would give tighter estimates.
2. **Dev subset is at a retrieval ceiling** — the ~98% hit rate says little about the full 1,025-Pokémon corpus. The manual full-data run is the real test of the agent loop.
3. **Faithfulness gap** — answers add details beyond the single retrieved document; retrieving more context or tightening the answer prompt would raise faithfulness.
4. **Single model** — all evaluations use the configured model (`MODEL_ID` from `.env`, required — no default). Results may differ with larger or different models.
5. **No end-to-end metric** — retrieval and answer quality were evaluated separately. A combined metric (e.g., answer correctness given retrieved context) would be more informative.

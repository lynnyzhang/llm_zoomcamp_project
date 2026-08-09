# Code Overview — How the Pieces Work Together

This report maps the modules and functions of the Pokémon RAG assistant and how
data flows through them. Read it top to bottom to see the full system; each
section notes the functions it calls into.

## Big picture

```
download_model ──► models/ (ONNX embedder)
ingest ──────────► data/corpus.jsonl
chunker ─────────► data/documents.jsonl ──► HybridSearch (index in memory)
                                                    │
                      ┌──────────────────────────────┤
                      ▼                              ▼
                 RAGBase (pipeline)          RAGAgent (agent loop)
                      │                              │
                      └──────────────► src/interface/app.py (Streamlit)
                                           │
                      ┌────────────────────┤
                      ▼                    ▼
              src/monitoring/        evaluation/
              tracer + dashboard     generate_qa + evals
```

Three layers: **data** (build the index), **runtime** (answer questions),
**observability** (measure both).

## 1. Data layer

| Module | Function | Job |
|---|---|---|
| `src/data/download_model.py` | `download()` | Fetch ONNX tokenizer + model into `models/` |
| `src/data/ingest.py` | `download_archive()`, `load_raw_pokedex()`, `parse_record()`, `main()` | Fetch Kaggle Pokémon dataset, cache the raw 1,025-record Pokédex, write one structured passage per Pokémon to `data/corpus.jsonl` (dev subset: first 50 by id) |
| `src/data/chunker.py` | `estimate_tokens()`, `split_passage()`, `generate_metadata()`, `process_corpus()`, `main()` | Split each corpus passage into documents with exact `id` linkage (`{pokemon_id}_{n}` never used — eval depends on pure ids), write `data/documents.jsonl` |

**Call chain:** `ingest.main()` writes `corpus.jsonl` → `chunker.main()` reads it
and writes `documents.jsonl`. Both are one-shot CLI scripts (`python -m
src.data.ingest`); the docker entrypoint runs them in sequence on boot.

## 2. Search layer

| Module | Function | Job |
|---|---|---|
| `src/search/embedder.py` | `Embedder.encode()`, `.encode_batch()` | Mean-pool + L2-normalize ONNX MiniLM embeddings (no torch) |
| `src/search/hybrid.py` | `HybridSearch.search()`, `.keyword_search()`, `.vector_search()`, `_rrf()` | Three modes: minsearch keyword, vector, and hybrid fused by Reciprocal Rank Fusion (`score = Σ w/(k+rank)`); `search_type` selects at call time |

**Call chain:** `RAGBase.search()` (below) dispatches on `search_type` to one of
the three `HybridSearch` methods.

## 3. Runtime layer (RAG)

| Module | Function | Job |
|---|---|---|
| `src/llm.py` | `get_api_key()`, `get_base_url()`, `get_model()`, `create_client()` | Central env config: API key, endpoint, model id, OpenAI client |
| `src/rag/pipeline.py` | `RAGBase.search()`, `.build_context()`, `.build_prompt()`, `.llm()`, `.rag()` | Plain pipeline: search → format context → prompt → LLM answer |
| `src/rag/agent.py` | `RAGAgent.run()`, `.perform_search()`, `.analyze_results()`, `.reformulate_query()`, `.generate_answer()`, `_is_out_of_scope()`, `_has_pokemon_signals()` | Agentic loop: up to `MAX_ITERATIONS=3` search/analyze/reformulate cycles; 3-layer guardrails (rule pre-gate, LLM off-topic flag, deterministic fail-safe) reject out-of-scope questions; low-confidence in-domain questions get `UNCERTAINTY_NOTE` instead |

**Call chain:** `app` → `RAGAgent.run(query)` → (guardrail check) →
`perform_search` → `HybridSearch.search` → `analyze_results` → either answer or
`reformulate_query` → repeat. `RAGBase.rag()` is the non-agentic path used by
evaluation for comparison.

## 4. Interface layer

| Module | Function | Job |
|---|---|---|
| `src/interface/app.py` | `_make_agent()`, `_maybe_trace()`, `render_message()`, `display_source_documents()`, `_pokemon_card_grid()`, `_record_feedback()`, `compute_confidence()` | Streamlit chat: agent answers + Pokémon cards with artwork, rejection banners for out-of-scope, thumbs up/down feedback |

**Call chain:** `render_message` is the single path for both history and live
replies; `display_source_documents` → `_unique_docs` → `_pokemon_card_grid`;
feedback goes to `record_feedback` (monitoring).

## 5. Monitoring layer

| Module | Function | Job |
|---|---|---|
| `src/monitoring/tracer.py` | `TracerSetup`, `SQLiteSpanExporter`, `PostgresSpanExporter`, `TracedRAGAgent.run()`, `.run_with_feedback()`, `record_feedback()`, `get_trace_stats()` | OpenTelemetry spans for every agent run/search/LLM call; SQLite always-on, Postgres dual-write when configured; feedback attaches to the exact `span_id` |
| `src/monitoring/dashboard.py` | `load_dataframe()`, `main()` | Streamlit dashboard over `traces.db` (fresh SQLite connection per query — thread-bound) |

**Call chain:** `app` wraps the agent in `TracedRAGAgent` → every `run()` emits
spans → exporters persist → `dashboard.py` reads back via `get_traces_db_path`.
`record_feedback(span_id, feedback)` updates the exact span row (SQLite
`UPDATE ... WHERE span_id = ?`, Postgres equivalent).

## 6. Evaluation layer

| Module | Function | Job |
|---|---|---|
| `evaluation/generate_qa.py` | `generate_for_record()`, `_generate_qa_pairs()`, `llm_structured_retry()`, `supports_structured_output()`, `main()` | LLM-generate 5 Q&A pairs per dev Pokémon (250 total) into `evaluation/data/qa.jsonl`; llama.cpp-compatible structured output via `patch_openai_client` |
| `evaluation/retrieval_eval.py` | `evaluate_search()`, `precision_at_k()`, `recall_at_k()`, `mrr()`, `main()` | Rank search quality per mode on the 250 questions |
| `evaluation/llm_eval.py` | `llm_judge()`, `llm_judge_retry()`, `evaluate_with_prompt()`, `main()` | LLM-as-judge (faithfulness/relevance/coherence, 1-5) across Simple / Detailed / With-Examples prompts |
| `evaluation/agent_eval.py` | `retrieval_accuracy()`, `llm_judge_score()`, `evaluate_answer_quality()`, `create_comparison_chart()`, `main()` | Simple RAG vs Agentic RAG: hit rate, searches/query, latency, judge scores, comparison chart |

**Call chain:** `generate_qa` feeds all three evals. Each eval writes JSON to
`evaluation/results/` and prints a summary; `agent_eval` also renders
`evaluation/results/agent_eval_comparison.png`. Results are committed per repo convention.

## Key integration points

1. **`HybridSearch`** is shared by `RAGBase`, `RAGAgent`, and `retrieval_eval` —
   the same index object serves the app and the evaluation scripts.
2. **`TracedRAGAgent`** wraps, not replaces, `RAGAgent` — monitoring is
   transparent to the RAG logic (a `RAGWithUsage`-style wrapper around
   `RAGBase`).
3. **Feedback round-trip:** UI → `record_feedback(span_id)` → span store →
   dashboard "feedback distribution" panel — the only user input that flows
   back into monitoring.
4. **Dev subset discipline:** `ingest --limit 50` and `generate_qa`'s corpus-id
   filtering keep every automated run on 50 docs / 250 QA; full-data runs are
   manual (`--full`), per user directive.
5. **Guardrail contract:** `RAGAgent` returns `rejected:true` with zero
   searches for out-of-scope queries (verified by tests asserting
   `searches == []`), and the app renders a warning banner instead of cards —
   the rejection behavior is the same in every layer.

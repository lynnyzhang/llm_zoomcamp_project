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
              monitoring/            evaluation/
              tracer + dashboard     generate_qa + evals
```

Three layers: **data** (build the index), **runtime** (answer questions),
**observability** (measure both).

## 1. Data layer

| Module | Function | Job |
|---|---|---|
| `src/data/download_model.py` | `download()` | Fetch ONNX tokenizer + model into `models/` |
| `src/data/ingest.py` | `download_archive()`, `extract_raw_csvs()`, `parse_row()`, `load_raw_rows()`, `build_corpus()`, `main()` | Fetch Kaggle Pokémon dataset (patelris/pokemon-dataset-with-stats-and-types), cache both raw CSVs (`pokemon_complete.csv`, `pokemon_types.csv`), write one structured record per Pokémon to `data/corpus.jsonl` (full dataset: all 1,350 by default; `--limit N` for a subset) |
| `src/data/chunker.py` | `load_type_chart()`, `build_evolution_map()`, `type_effectiveness()`, `build_search_text()`, `build_pokemon_doc()`, `type_chart_doc()`, `main()` | Build one Pokémon-native document per record with pure-int `id` linkage (eval depends on exact ids), derive `type_effectiveness` (18 multipliers from the type chart) and `evolves_from`/`evolves_into` (chain id + dex order), append 18 type-chart documents, write `data/chunks/documents.jsonl` |

**Call chain:** `ingest.main()` writes `corpus.jsonl` → `chunker.main()` reads it
and writes `documents.jsonl`. Both are one-shot CLI scripts (`python -m
src.data.ingest`); the docker entrypoint runs them in sequence on boot.

## 2. Search layer

| Module | Function | Job |
|---|---|---|
| `src/search/embedder.py` | `Embedder.encode()`, `.encode_batch()` | Mean-pool + L2-normalize ONNX MiniLM embeddings (no torch) |
| `src/search/hybrid.py` | `HybridSearch.search()`, `.keyword_search()`, `.vector_search()`, `reciprocal_rank_fusion()` | Three modes: minsearch keyword, vector, and hybrid fused by Reciprocal Rank Fusion (`score = Σ w/(k+rank)`); `search_type` selects at call time |

**Call chain:** `RAGBase.search()` (below) dispatches on `search_type` to one of
the three `HybridSearch` methods.

## 3. Runtime layer (RAG)

| Module | Function | Job |
|---|---|---|
| `src/llm.py` | `LLMClient` (`get_api_key()`, `get_base_url()`, `get_model()`, `get()`) | Central env config: API key, endpoint, model id, OpenAI client wrapper |
| `src/rag/RAGBase.py` | `RAGBase.search()`, `.build_context()`, `.build_prompt()`, `.llm()`, `.rag()` | Plain pipeline: search → format context → prompt → LLM answer |
| `src/rag/agent.py` | `RAGAgent.run()`, `.perform_search()`, `build_graph()`, `local_search()`, `local_judge()`, `web_search_node()`, `web_judge()`, `answer_node()`, `reject_node()`, `fallback_node()`, `finalize_node()`, `judge()` | LangGraph escalate flow: local hybrid search always first → LLM judge (verdict + confidence) → Bulbapedia web search (Tavily, `src/search/web.py`) only when local results are insufficient; only confident answers are returned (gated by `CONFIDENCE_THRESHOLD`, default 0.7), otherwise the rejection message |

**Call chain:** `app` → `RAGAgent.run(query)` → `local_search` (HybridSearch)
→ `local_judge` → `answer_node` | (`web_search` → `web_judge` → `answer_node` /
`reject_node`) → `finalize`. `RAGBase.rag()` is the non-agentic path used by
evaluation for comparison.

## 4. Interface layer

| Module | Function | Job |
|---|---|---|
| `src/interface/app.py` | `make_agent()`, `maybe_trace()`, `render_message()`, `render_message_body()`, `filter_docs_by_question()`, `pokemon_card_grid()`, `record_feedback()`, `parse_cli_flags()` | Streamlit chat: agent answers with source caption (local knowledge base / Bulbapedia), optional LLM-judge confidence bar (`--show-confidence` launch flag), Pokémon cards (only Pokémon named in the question) with artwork, rejection banners, thumbs up/down feedback |

**Call chain:** `render_message` is the single path for both history and live
replies; `render_message_body` → `unique_docs` → `filter_docs_by_question` →
`pokemon_card_grid`; feedback goes to `record_feedback` (monitoring).

## 5. Monitoring layer

| Module | Function | Job |
|---|---|---|
| `monitoring/tracer.py` | `TracerSetup`, `SQLiteSpanExporter`, `PostgresSpanExporter`, `TracedRAGAgent.run()`, `.run_with_feedback()`, `record_feedback()`, `get_trace_stats()` | OpenTelemetry spans for every agent run/search/LLM call; SQLite always-on, Postgres dual-write when configured; feedback attaches to the exact `span_id` |
| `monitoring/dashboard.py` | `load_dataframe()`, `main()` | Streamlit dashboard over `monitoring/traces.db` (fresh SQLite connection per query — thread-bound) |

**Call chain:** `app` wraps the agent in `TracedRAGAgent` → every `run()` emits
spans → exporters persist → `dashboard.py` reads back via `get_traces_db_path`.
`record_feedback(span_id, feedback)` updates the exact span row (SQLite
`UPDATE ... WHERE span_id = ?`, Postgres equivalent).

## 6. Evaluation layer

| Module | Function | Job |
|---|---|---|
| `evaluation/evaluation_utils.py` | `calc_price()`, `calc_total_price()`, `patch_openai_client()`, `llm_structured()`, `llm_structured_retry()`, `map_progress()`, `load_document_index()`, `ground_truth_answer()` | Shared eval helpers mirroring the course's `evaluation_utils.py`; the document index provides ground-truth answers (`search_text`) at eval time |
| `evaluation/generate_qa.py` | `generate_questions_for_record()`, `generate_questions()`, `supports_structured_output()`, `main()` | LLM-generate 5 questions per Pokémon linked to the document containing the answer (default dev subset: coverage-sampled 50 → 250 rows of `{"question", "document"}`; `--full` for all) into `evaluation/data/qa.jsonl`; llama.cpp-compatible structured output via shared `patch_openai_client` |
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
4. **Dev subset discipline:** `ingest.py` builds the full 1,350-record corpus by default; `generate_qa.py`'s coverage-sampled dev subset (50 records → 250 questions) keeps every automated run cheap; full-data QA runs are manual (`--full`), per user directive.
5. **Ground truth = question → document:** `generate_qa.py` writes only questions (`{"question", "document"}`); the LLM never writes answers — `llm_eval`/`agent_eval` resolve the ground-truth answer from the linked document's `search_text`, keeping the judge honest.
6. **Guardrail contract:** `RAGAgent` returns `rejected:true` with the
   rejection message when the LLM judge finds no confident answer from local
   or Bulbapedia search (out-of-scope, below-threshold, or web failure), and
   the app renders a warning banner instead of cards — the rejection behavior
   is the same in every layer.

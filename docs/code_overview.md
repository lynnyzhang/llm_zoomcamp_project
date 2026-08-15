# Code Overview — How the Pieces Work Together

This report maps the modules and functions of the Pokémon RAG assistant and how
data flows through them. Read it top to bottom to see the full system; each
section notes the functions it calls into.

## Big picture

```
download_model ──► models/ (ONNX embedder)
ingest ──────────► data/pokemon.jsonl
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
| `src/data/ingest.py` | `download_archive()`, `extract_raw_csvs()`, `parse_row()`, `load_raw_rows()`, `build_dataset()`, `main()` | Fetch Kaggle Pokémon dataset (patelris/pokemon-dataset-with-stats-and-types), cache both raw CSVs (`pokemon_complete.csv`, `pokemon_types.csv`), write one structured record per Pokémon to `data/pokemon.jsonl` (full dataset: all 1,350 by default; `--limit N` for a subset) |
| `src/data/chunker.py` | `load_type_chart()`, `build_evolution_map()`, `type_effectiveness()`, `build_search_text()`, `build_pokemon_doc()`, `type_chart_doc()`, `main()` | Build one Pokémon-native document per record with pure-int `id` linkage (eval depends on exact ids), derive `type_effectiveness` (18 multipliers from the type chart) and `evolves_from`/`evolves_into` (chain id + dex order), append 18 type-chart documents, write `data/chunks/documents.jsonl` |

**Call chain:** `ingest.main()` writes `pokemon.jsonl` → `chunker.main()` reads it
and writes `documents.jsonl`. Both are one-shot CLI scripts (`python -m
src.data.ingest`); the docker entrypoint runs them in sequence on boot.

## 2. Search layer

| Module | Function | Job |
|---|---|---|
| `src/search/embedder.py` | `Embedder.encode()`, `.encode_batch()` | Mean-pool + L2-normalize ONNX MiniLM embeddings (no torch) |
| `src/search/hybrid.py` | `HybridSearch.search()`, `.keyword_search()`, `.vector_search()`, `rrf()` | Hybrid search = minsearch keyword + vector results fused by Reciprocal Rank Fusion (`score = Σ 1/(k+rank)`); the standalone keyword/vector methods serve `retrieval_eval.py` |

**Call chain:** `RAGBase.search()` (below) runs the hybrid search, then
the three `HybridSearch` methods.

## 3. Runtime layer (RAG)

| Module | Function | Job |
|---|---|---|
| `src/llm.py` | `LLMClient` (`get_api_key()`, `get_base_url()`, `get_model()`, `get()`) | Central env config: API key, endpoint, model id, OpenAI client wrapper |
| `src/rag/RAGBase.py` | `RAGBase.search()`, `.build_context()`, `.build_prompt()`, `.llm()`, `.rag()` | Plain pipeline: search → format context → prompt → LLM answer |
| `src/rag/agent.py` | `RAGAgent.run()`, `.perform_search()`, `.execute_tool()`, `.format_tool_results()`, `.finalize()`, `cosine_similarity()` | Manual LLM tool-use loop (no graph framework): the model decides when to call `search_local_knowledge_base` / `search_bulbapedia` (Tavily, `src/search/web.py`) and writes its own web keyword queries; tool results feed back as `function_call_output` items until the model replies with a final answer; the final answer is gated by a programmatic grounding score (max embedding-cosine between the answer and any retrieved record, gated by `CONFIDENCE_THRESHOLD`, default 0.65) and otherwise replaced by the rejection message; `result["relevance"]` reports the question-answer embedding cosine; loop bounded by `MAX_ITERATIONS` |

**Call chain:** `app` → `RAGAgent.run(query)` → `local_search` (HybridSearch)
→ `local_judge` → `answer_node` | (`web_search` → `web_judge` → `answer_node` /
`reject_node`) → `finalize`. `RAGBase.rag()` is the non-agentic path used by
evaluation for comparison.

## 4. Interface layer

| Module | Function | Job |
|---|---|---|
| `src/interface/app.py` | `make_agent()`, `maybe_trace()`, `render_message()`, `render_message_body()`, `pokemon_doc()`, `pokemon_card_grid()`, `record_feedback()`, `parse_cli_flags()` | Streamlit chat: agent answers with source caption (local knowledge base / Bulbapedia), optional grounding-confidence bar (`--show-confidence` launch flag), Pokémon cards (only Pokémon named in the question) with artwork, rejection banners, thumbs up/down feedback |

**Call chain:** `render_message` is the single path for both history and live
replies; `render_message_body` → `pokemon_doc` →
`pokemon_card_grid`; feedback goes to `record_feedback` (monitoring).

## 5. Monitoring layer

| Module | Function | Job |
|---|---|---|
| `monitoring/db_init.py` | `get_db_connection()`, `init_db()`, `init_feedback()` | psycopg connection to the capstone Postgres (env-configurable host/port/db/user/password) and idempotent schema creation |
| `monitoring/metrics.py` | `LLMCallRecord`, `calculate_cost()` | Course-shaped per-call record (model, tokens, response time, cost, source, rejection flag, span_id) and cost from usage (qwen pricing) |
| `monitoring/tracer.py` | `TracerSetup`, `PostgresSpanExporter`, `TracedRAGAgent.run()`, `.run_with_feedback()`, `record_feedback()`, `get_trace_stats()`, `tracing_enabled()` | OpenTelemetry spans for every agent run/search/LLM call, stored in the same Postgres DB; feedback attaches to the exact `span_id` |
| `monitoring/db_save.py` | `save_conversation()`, `save_search()`, `save_llm_call()` | Persist one `LLMCallRecord` per assistant turn (question, answer, model, tokens, cost, response time, source, rejection flag, span_id, error) plus per-search results (JSON) and per-LLM-call usage/latency/error to Postgres |
| `monitoring/db_feedback.py` | `save_feedback()` | Persist course-shaped user feedback (source, score, relevance, explanation) keyed by conversation id |
| `monitoring/db_query.py` | `get_conversations()`, `get_stats()`, `get_user_feedback_stats()`, `get_feedback_for_conversations()` | Read conversations (optionally session-scoped), aggregate stats, and feedback for the UI and dashboard; never raises, safe defaults |
| `monitoring/dashboard.py` | `load_dataframe()`, `main()` | Streamlit dashboard over the Postgres store (fresh psycopg connection per query — thread-bound) |

**Call chain:** `app` wraps the agent in `TracedRAGAgent` → every `run()` emits
spans → `PostgresSpanExporter` persists to Postgres; each turn is also saved as
a conversation via `db_save.save_conversation`, tagged with a per-browser-session
`session_id` (a uuid kept in `st.session_state`) and an `error` field (failed
turns are persisted as rejected rows for back-trace, not shown in chat history).
The turn's per-search results and per-LLM-call usage are written to the
`searches` and `llm_calls` tables via `db_save.save_search` and
`save_llm_call`. The UI restores the current session's recent history via
`db_query.get_conversations(session_id=...)`, and `dashboard.py` reads spans
through `load_dataframe` and conversations through `db_query`.
`record_feedback(span_id, feedback)` updates the exact span row (Postgres
`UPDATE ... WHERE span_id = ?`) and the button also writes a course-shaped
feedback row via `db_feedback.save_feedback`. All monitoring data — spans,
conversations, searches, llm_calls, and feedback — lives in the same Postgres
database.

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
3. **Feedback round-trip:** UI → `record_feedback(span_id)` (span store, feeds
   the dashboard "feedback distribution" panel) and `save_feedback` (course-shaped
   feedback row, feeds the dashboard "User feedback" thumbs metrics) — the user
   input that flows back into monitoring.
4. **Dev subset discipline:** `ingest.py` builds the full 1,350-record dataset by default; `generate_qa.py`'s coverage-sampled dev subset (50 records → 250 questions) keeps every automated run cheap; full-data QA runs are manual (`--full`), per user directive.
5. **Ground truth = question → document:** `generate_qa.py` writes only questions (`{"question", "document"}`); the LLM never writes answers — `llm_eval`/`agent_eval` resolve the ground-truth answer from the linked document's `search_text`, keeping the judge honest.
6. **Guardrail contract:** `RAGAgent` returns `rejected:true` with the
   rejection message when the model refuses (out-of-scope), the grounding
   gate fails (below `CONFIDENCE_THRESHOLD`), or the loop is exhausted, and
   the app renders a warning banner instead of cards — the rejection behavior
   is the same in every layer.

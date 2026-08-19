# AGENTS.md — LLM Zoomcamp 2026 Capstone

## What This Is

The capstone application for the DataTalksClub LLM Zoomcamp 2026 course: an
agentic RAG assistant for Pokémon knowledge with a Streamlit chat UI, hybrid
search (keyword + vector, RRF fusion), LLM-driven tool use (local + Bulbapedia search),
out-of-scope guardrails, evaluation scripts, monitoring, and Docker
deployment.

The knowledge base is the Kaggle Pokémon Dataset with Stats and Types
(`patelris/pokemon-dataset-with-stats-and-types`, 1,350 records: 1,025
canonical Pokémon + 325 alternate forms) fetched by `src/data/build_documents.py`,
which builds the full 1,350-record dataset by default. The **dev subset** — a
deterministic coverage-sampled 50 Pokémon (250 ground-truth questions) — is the default for
ground-truth generation (`evaluation/data/src/generate_qa.py`) and all automated eval
runs; full-data QA runs are manual (see `docs/setup.md`).

**Self-contained:** this project depends only on its own `src/` — never import
from external course content or reference material.

**Reference folders:** a set of numbered course-material folders at the repo
root (e.g. `<number>-<Function>`) are USER-PROVIDED reference only. They are
untracked (ignored by the repo `.gitignore` and a global gitignore rule), never
imported by `src/`, never staged or committed, and never named in project
documents. The user removes them; do not delete, move, or add them.

**Design guidance:** mirror the reference folders' code design as closely as
possible — reuse their patterns unless a production limitation (scale,
reliability, observability, maintainability) requires improvement.

**Course parity for math and signatures:** mathematical functions and class
signatures must match the numbered course reference exactly unless a
deviation is justified by test results or a documented production
limitation. Prefer the course's own names, formulas, constants, and default
values (e.g. `rrf`, `k=60`, the `text_format`/`response_format` patch
pattern) over "improved" variants with no measured benefit.

**Production complexity vs dead parameters:** production code is expected to
be more complex than course code because edge cases must be handled — that
complexity is earned. Unused parameters are not complexity, they are dead
code. A parameter must have a real consumer to exist: a production caller,
an env-backed configuration use case, or a course-identical signature.
Consumption only by tests does NOT justify a feature parameter (test seams
for infrastructure — e.g. `db_path` for tmp-dir isolation — are the
exception). When auditing, classify parameters into: real production use /
course parity / test seam / dead — and remove the last category.

## Code Organization

- **Layers, one concern per file** — `src/` owns domain logic,
  `monitoring/` owns persistence/tracing, `src/interface/` owns UI. No
  mixing: no SQL in UI code, no UI layout in domain classes, no record
  construction in the UI layer.
- **Encapsulation** — objects own their state; callers read it, never
  mutate it from outside. State changes go through the owning object's
  methods (e.g. the agent's records are updated by the agent, not by the
  app).
- **File cap** — no implementation file exceeds 200 lines. When a file
  grows past it, split by responsibility: one class per concern, one
  module per layer. Do not grow the file.
- **Records live beside their domain** — value objects (LLMCallRecord,
  SearchRecord) sit with the classes that produce them, not in the layer
  that persists them.
- **Inheritance vs composition** — prefer composition; use inheritance
  when a class is genuinely a specialization (RAGAgent is-a RAGBase).

## Code Style

- **No "what" comments** — code should be self-explanatory: keep functions
  short and simple, and name variables to reflect their purpose. Do not add
  comments describing what the code does.
- **Most common word naming** — name variables and functions with the most
  commonly used word for the meaning (e.g. `test` instead of `probe`).
- **One word, one meaning** — never reuse the same word for different
  concepts in the same codebase (e.g. "turn" once meant a whole agent loop,
  a single LLM API call, and a scripted test response). When a name becomes
  ambiguous, disambiguate by renaming each use to its precise meaning
  (`agent_loop`, `llm_api_call`, `response`).
- **Plain names for files and folders** — file and folder names must also
  use plain, everyday words (e.g. `data/chunks/documents.jsonl` not
  `data/corpus.jsonl`). This project is a demo; anyone should be able to
  understand it at a glance.
- **Files named after their main class or function** — when a module has a
  single main class or function, the file name is its snake_case name
  (e.g. `message_renderer.py` for `MessageRenderer`, `web_search.py` for
  `web_search()`). This is the Java/Ruby convention applied to Python:
  `Foo.java` holds `Foo`, `message_renderer.py` holds `MessageRenderer`. Vague
  multi-purpose names (`utils.py`, `records.py`, `metrics.py`) are
  acceptable only when the module genuinely holds several co-equal pieces
  (`scoring.py`, `prompts.py`); otherwise they hide what the file does.
- **Comments only for "why"** — add comments solely for edge cases,
  workarounds, and patches, explaining why they exist. Minimize comment
  presence.
- **API documentation only when mandatory** — add a doc comment only where
  it is required for understanding (e.g. a function the LLM needs to
  understand what it does). Use the language's native doc-comment
  convention (docstrings in Python, JSDoc in JS/TS, doc comments in
  Go/Rust/Swift/Kotlin, Javadoc in Java).
- **No leading-underscore names** — never prefix variables, functions,
  methods, classes, or module-level constants with a single underscore
  (e.g. write `judge` not `_judge`, `DB_PATH` not `_DB_PATH`). This
  project does not use Python's private-marking convention; drop the
  underscore everywhere (Rust's `_unused` idiom is the exception).

## Setup

```bash
uv sync                          # install deps into .venv (Python 3.13+)
cp .env.example .env             # then edit the LLM vars (see below)
uv run python -m src.data.download_model   # fetch ONNX embedder artifacts
uv run python -m src.data.build_documents   # build chunks/documents.jsonl (full 1,350; --limit N for a subset)
uv run streamlit run src/interface/app.py  # dev app on :8501
```

**Env vars are documented in two places: `.env.example` (values + comments)
and `docs/setup.md` (reference table). Every new env var must be added to
both — never add a var the user cannot discover.**

Docker alternative: `docker-compose up --build` (app + Postgres + Grafana).

## LLM Backend

`.env` supplies the config, read centrally by `src/llm_client.py` (which loads
`src/llm/env.py`): `LLMClient.get_api_key()`, `LLMClient.get_base_url()`,
`LLMClient.get_model()`, `LLMClient.get()`.

- `OPENAI_API_KEY` — API key (required; RuntimeError if missing)
- `OPENAI_API_BASE_URL` (or legacy `OPENAI_BASE_URL`) — the OpenAI-compatible
  endpoint: a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud
  API (e.g. `https://api.openai.com/v1`); RuntimeError if missing
- `MODEL_ID` — model name (required, no default fallback; RuntimeError if
  missing)
- `DATASET_PATH` — optional local data directory override (default `./data`,
  used by `src/data/build_documents.py`)
- `TAVILY_API_KEY` — web search backend for the agent's escalation path
  (https://tavily.com); missing/empty key makes web escalation return empty
  results so the agent rejects gracefully
- `CONFIDENCE_THRESHOLD` — minimum grounding score (0..1) an answer must have to
  be returned (faithfulness: max embedding-cosine similarity to any retrieved
  record); below it the answer is replaced by the rejection message. Default 0.65
- `RETRIEVAL_SCORE_THRESHOLD` — minimum query↔chunk cosine (0..1) a retrieved
  chunk must have to be returned by hybrid search; below it the chunk is dropped
  as irrelevant. Default 0.3
- `AGENT_TEMPERATURE` — sampling temperature for the agent tool-use loop.
  Default 0.0 (deterministic); set to 1 for reasoning models (o1/o3)
- `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` / `POSTGRES_USER` /
  `POSTGRES_PASSWORD` — monitoring store (conversations, spans, searches,
  llm_calls, feedback); used by the app, dashboards, and docker-compose
- `TRACING_ENABLED` — OpenTelemetry span export (spans land in Postgres);
  default enabled ("1"), set to "0" to disable

All LLM calls use the OpenAI Responses API (`client.responses.create`), not
Chat Completions. In Docker, `deployment/entrypoint.sh` rewrites a
`localhost`/`127.0.0.1` base URL to `host.docker.internal` so the container
can reach a locally hosted LLM on the host (cloud URLs pass through
unchanged).

## Key Libraries

- **minsearch** — keyword search index (NOT vector search despite the name);
  keyword half of hybrid search (`src/search/hybrid_search.py`)
- **onnxruntime + tokenizers** — local embeddings (all-MiniLM-L6-v2 ONNX,
  no torch) via `src/search/embedder.py`
- **openai** — LLM calls (Responses API)
- **streamlit** — chat UI (`src/interface/app.py`) + monitoring dashboards
- **opentelemetry** — tracing with Postgres storage (`monitoring/`)
- **grafana** — dashboards on top of the Postgres span store (`monitoring/dashboards/`)

## Structure

| Path | Purpose |
|------|---------|
| `src/llm_client.py` | env config + OpenAI client wrapper: `LLMClient.get_api_key()`, `get_base_url()`, `get_model()`, `get()` |
| `src/llm/` | `env.py` — env var readers (api key, base URL, model, agent temperature); loaded by `src/llm_client.py` via importlib |
| `src/data/` | `build_documents.py` (corpus entry point), `download_model.py`, `csv_parsers.py`, `download.py`, `evolution.py`, `evolution_overrides.py`, `type_chart.py`, `pokemon_doc_builder.py`, `chunking.py` |
| `src/search/` | `hybrid_search.py` (keyword + vector + RRF), `embedder.py` (ONNX), `web_search.py` (Tavily), `search_records.py`, `reranker.py` |
| `src/rag/` | `rag_base.py` (RAGBase), `rag_agent.py` (RAGAgent, manual agentic loop: LLM tool calls, guardrails), `llm_call_record.py` (LLMCallRecord + cost), `scoring.py`, `tools.py` (tool defs + SearchRecord + execution), `prompts.py` |
| `src/interface/` | `app.py` (Streamlit entry), `chat_page.py` (ChatPage), `chat_message.py` (ChatMessage), `message_renderer.py` (MessageRenderer), `agent_loop_saver.py` (AgentLoopSaver) |
| `evaluation/` | offline eval (pre-deployment). `data/` holds `qa.jsonl` (dev-subset ground-truth questions) and `gate_collection.jsonl` (judged gate data), plus `src/` scripts: `generate_qa.py` (QA generation entry point), `qa_generation.py` (question generation + dev-subset sampling), `collect_gate_data.py` (judged gate-data collection). `notebooks/` holds the committed `01_agent_path_analysis.ipynb`…`05_answer_quality.ipynb`, `share/` (shared eval code: `common.py`, `document_index.py`, `judge_prompts.py`, `llm_calls.py`, `judge.py` — consolidated correctness + grounding judges), and `results/` (committed notebook outputs). |
| `monitoring/` | `tracer.py` (TracerSetup + TracedRAGAgent), `span_exporter.py` (PostgresSpanExporter), `span_store.py`, `db_init.py` (connection + schema + init), `db_save.py` (save_conversation, save_search, save_llm_call), `db_feedback.py` (save_feedback), `db_query.py`, `db_stats.py`; `grafana/` + `dashboards/` configs on the same Postgres |
| `tests/` | pytest suite (`conftest.py`, `test_integration.py`, `test_llm.py`) |
| `docs/` | `setup.md`, `usage.md`, `evaluation.md` |
| `deployment/` | `Dockerfile`, `.dockerignore`, `entrypoint.sh` (pipeline orchestration + URL rewrite) |
| root config | `docker-compose.yml` (entry point; stays at root), `pyproject.toml`, `uv.lock`, `.python-version`, `project.md`, `README.md`, `.gitignore` |
| `monitoring/grafana` + `monitoring/dashboards` | Grafana provisioning + dashboard JSON on the Postgres span store |

## Testing

```bash
set -a; source .env; set +a; uv run pytest -q   # 124 tests — keep this count in sync with the suite
```

## Gotchas

- **`.env` is required for any LLM call** — `LLMClient.get_model()` raises without
  `MODEL_ID`; there is no default model. LLM calls fail lazily at first use.
- **`.env` must never be committed** — it holds the LLM API key and is ignored by the repo `.gitignore` (and a global gitignore rule).
- `uv.lock` and `.python-version` are committed in this repo; `evaluation/notebooks/results/` eval outputs are committed too.
- `data/` and `models/` hold downloaded/generated artifacts — populated by the setup commands above (`data/chunks/documents.jsonl`, ONNX embedder under `models/`); the two raw CSVs under `data/raw/` are **bundled and committed** (Kaggle anonymous downloads are bot-blocked, so the repo ships its own copy — no login needed); `evaluation/data/qa.jsonl` is an LLM-generated eval artifact; all monitoring data (spans + conversations + searches + llm_calls + feedback) lives in Postgres (`docker-compose up postgres`, or a local server on localhost:5432 with the capstone defaults; set `POSTGRES_HOST` etc. to override).
- Keep the project self-contained: no imports from external reference
  material (docstring attributions are comments only, never dependencies).

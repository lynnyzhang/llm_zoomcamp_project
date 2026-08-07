# AGENTS.md — LLM Zoomcamp 2026 Capstone

## What This Is

The capstone application for the DataTalksClub LLM Zoomcamp 2026 course: an
agentic RAG assistant for Pokémon knowledge with a Streamlit chat UI, hybrid
search (keyword + vector, RRF fusion), LLM-driven query reformulation,
out-of-scope guardrails, evaluation scripts, monitoring, and Docker
deployment.

The knowledge base is the Kaggle Complete Pokémon Dataset
(`elroytan/pokemondata`, 1,025 records) fetched by `src/data/ingest.py`. The
**dev subset** (first 50 Pokémon by id, 250 QA pairs) is the default for all
automated runs; full-data runs are manual (see `docs/setup.md`).

**Self-contained:** this project depends only on its own `src/` — never import
from external course content or reference material. (A few source files carry
docstring attributions like "Adapted from 4-Evaluation/rag_helper.py" —
comments only, not dependencies.)

**Reference folders:** `4-Evaluation/` and `5-Monitoring/` at the repo root are
USER-PROVIDED reference folders from the course material. They are untracked
(via a global gitignore rule — the repo has no `.gitignore`), never imported by
`src/`, and never staged or committed. The user removes them; do not delete,
move, or add them.

## Setup

```bash
uv sync                          # install deps into .venv (Python 3.13+)
cp .env.example .env             # then edit the LLM vars (see below)
uv run python -m src.data.download_model   # fetch ONNX embedder artifacts
uv run python -m src.data.ingest           # fetch Kaggle dataset, build corpus.jsonl (dev subset: 50)
uv run python -m src.data.chunker          # build chunks/documents.jsonl (indexed)
uv run streamlit run src/interface/app.py  # dev app on :8501
```

Docker alternative: `docker-compose up --build` (app + Postgres + Grafana).

## LLM Backend

`.env` supplies the LLM config, read centrally by `src/llm.py`
(`get_api_key()`, `get_base_url()`, `get_model()`, `create_client()`):

- `OPENAI_API_KEY` — API key (required; RuntimeError if missing)
- `OPENAI_API_BASE_URL` (or legacy `OPENAI_BASE_URL`) — the OpenAI-compatible
  endpoint: a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud
  API (e.g. `https://api.openai.com/v1`); RuntimeError if missing
- `MODEL_ID` — model name (required, no default fallback; RuntimeError if
  missing)

All LLM calls use the OpenAI Responses API (`client.responses.create`), not
Chat Completions. In Docker, `docker/entrypoint.sh` rewrites a
`localhost`/`127.0.0.1` base URL to `host.docker.internal` so the container
can reach a locally hosted LLM on the host (cloud URLs pass through
unchanged).

## Key Libraries

- **minsearch** — keyword search index (NOT vector search despite the name);
  keyword half of hybrid search (`src/search/hybrid.py`)
- **onnxruntime + tokenizers** — local embeddings (all-MiniLM-L6-v2 ONNX,
  no torch) via `src/search/embedder.py`
- **openai** — LLM calls (Responses API)
- **streamlit** — chat UI (`src/interface/app.py`) + monitoring dashboards
- **opentelemetry** — tracing with SQLite/Postgres storage (`src/monitoring/`)
- **grafana** — dashboards on top of the Postgres span store (`dashboards/`)

## Structure

| Path | Purpose |
|------|---------|
| `src/llm.py` | env config: API key, base URL, model ID, client creation |
| `src/data/` | `download_model.py`, `chunker.py`, `ingest.py` (index build) |
| `src/search/` | `hybrid.py` (keyword + vector + RRF), `embedder.py` (ONNX) |
| `src/rag/` | `pipeline.py` (RAGBase), `agent.py` (agentic loop + guardrails) |
| `src/interface/app.py` | Streamlit chat UI (Pokémon cards, feedback) |
| `src/evaluation/` | `generate_qa.py` (QA set), `retrieval_eval.py`, `llm_eval.py`, `agent_eval.py` |
| `src/monitoring/` | tracer (SQLiteSpanExporter + PostgresSpanExporter) + Grafana dashboards |
| `tests/` | pytest suite (`conftest.py`, `test_integration.py`) |
| `docker/` | `entrypoint.sh` (pipeline orchestration + URL rewrite) |

## Testing

```bash
set -a; source .env; set +a; uv run pytest -q   # 116 tests
```

## Gotchas

- **`.env` is required for any LLM call** — `get_model()` raises without
  `MODEL_ID`; there is no default model. LLM calls fail lazily at first use.
- **`.env` must never be committed** — it holds the LLM API key (ignored via a global gitignore rule; no repo `.gitignore` exists — add one).
- `uv.lock` and `.python-version` are committed in this repo; `results/` eval outputs are committed too.
- `data/` and `models/` hold downloaded/generated artifacts — currently empty here; fetch them with `uv run python -m src.data.download_model`.
- Keep the project self-contained: no imports from external reference
  material (docstring attributions are comments only, never dependencies).

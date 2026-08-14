# AGENTS.md — LLM Zoomcamp 2026 Capstone

## What This Is

The capstone application for the DataTalksClub LLM Zoomcamp 2026 course: an
agentic RAG assistant for Pokémon knowledge with a Streamlit chat UI, hybrid
search (keyword + vector, RRF fusion), LLM-driven tool use (local + Bulbapedia search),
out-of-scope guardrails, evaluation scripts, monitoring, and Docker
deployment.

The knowledge base is the Kaggle Pokémon Dataset with Stats and Types
(`patelris/pokemon-dataset-with-stats-and-types`, 1,350 records: 1,025
canonical Pokémon + 325 alternate forms) fetched by `src/data/ingest.py`,
which builds the full 1,350-record dataset by default. The **dev subset** — a
deterministic coverage-sampled 50 Pokémon (250 ground-truth questions) — is the default for
ground-truth generation (`evaluation/generate_qa.py`) and all automated eval
runs; full-data QA runs are manual (see `docs/setup.md`).

**Self-contained:** this project depends only on its own `src/` — never import
from external course content or reference material.

**Reference folders:** a set of numbered course-material folders at the repo
root (e.g. `<number>-<Function>`) are USER-PROVIDED reference only. They are
untracked (via a global gitignore rule — the repo has no `.gitignore`), never
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

## Code Style

- **No "what" comments** — code should be self-explanatory: keep functions
  short and simple, and name variables to reflect their purpose. Do not add
  comments describing what the code does.
- **Most common word naming** — name variables and functions with the most
  commonly used word for the meaning (e.g. `test` instead of `probe`).
- **Plain names for files and folders** — file and folder names must also
  use plain, everyday words (e.g. `data/pokemon.jsonl` not
  `data/corpus.jsonl`). This project is a demo; anyone should be able to
  understand it at a glance.
- **Comments only for "why"** — add comments solely for edge cases,
  workarounds, and patches, explaining why they exist. Minimize comment
  presence.
- **Docstrings only when mandatory** — add a docstring only where it is
  required for understanding (e.g. a function the LLM needs to understand
  what it does).
- **No leading-underscore names** — never prefix variables, functions,
  methods, classes, or module-level constants with a single underscore
  (e.g. write `judge` not `_judge`, `DB_PATH` not `_DB_PATH`). This
  project does not use Python's private-marking convention; drop the
  underscore everywhere.

## Setup

```bash
uv sync                          # install deps into .venv (Python 3.13+)
cp .env.example .env             # then edit the LLM vars (see below)
uv run python -m src.data.download_model   # fetch ONNX embedder artifacts
uv run python -m src.data.ingest           # fetch Kaggle dataset, build pokemon.jsonl (dev subset: 50)
uv run python -m src.data.chunker          # build chunks/documents.jsonl (indexed)
uv run streamlit run src/interface/app.py  # dev app on :8501
```

Docker alternative: `docker-compose up --build` (app + Postgres + Grafana).

## LLM Backend

`.env` supplies the LLM config, read centrally by `src/llm.py`
(`LLMClient.get_api_key()`, `LLMClient.get_base_url()`, `LLMClient.get_model()`, `LLMClient.get()`):

- `OPENAI_API_KEY` — API key (required; RuntimeError if missing)
- `OPENAI_API_BASE_URL` (or legacy `OPENAI_BASE_URL`) — the OpenAI-compatible
  endpoint: a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud
  API (e.g. `https://api.openai.com/v1`); RuntimeError if missing
- `MODEL_ID` — model name (required, no default fallback; RuntimeError if
  missing)
- `DATASET_PATH` — optional local data directory override (default `./data`,
  used by `src/data/ingest.py`)

All LLM calls use the OpenAI Responses API (`client.responses.create`), not
Chat Completions. In Docker, `deployment/entrypoint.sh` rewrites a
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
- **opentelemetry** — tracing with SQLite/Postgres storage (`monitoring/`)
- **grafana** — dashboards on top of the Postgres span store (`monitoring/dashboards/`)

## Structure

| Path | Purpose |
|------|---------|
| `src/llm.py` | env config: API key, base URL, model ID, client creation |
| `src/data/` | `download_model.py`, `chunker.py`, `ingest.py` (index build) |
| `src/search/` | `hybrid.py` (keyword + vector + RRF), `embedder.py` (ONNX) |
| `src/rag/` | `RAGBase.py` (RAGBase), `agent.py` (manual agentic loop: LLM tool calls, guardrails) |
| `src/interface/app.py` | Streamlit chat UI (Pokémon cards, feedback) |
| `evaluation/` | offline eval (pre-deployment): `generate_qa.py` (QA set), `retrieval_eval.py`, `llm_eval.py`, `agent_eval.py`; `data/` (qa.jsonl), `results/` |
| `monitoring/` | `tracer.py` (SQLiteSpanExporter + PostgresSpanExporter), `dashboard.py` (Streamlit); runtime `traces.db`; `grafana/` + `dashboards/` configs |
| `tests/` | pytest suite (`conftest.py`, `test_integration.py`) |
| `docs/` | `setup.md`, `usage.md`, `evaluation.md`, `code_overview.md` |
| `deployment/` | `Dockerfile`, `.dockerignore`, `entrypoint.sh` (pipeline orchestration + URL rewrite) |
| root config | `docker-compose.yml` (entry point; stays at root), `project.md`, `README.md` |
| `monitoring/grafana` + `monitoring/dashboards` | Grafana provisioning + dashboard JSON on the Postgres span store |

## Testing

```bash
set -a; source .env; set +a; uv run pytest -q   # 115 tests
```

## Gotchas

- **`.env` is required for any LLM call** — `LLMClient.get_model()` raises without
  `MODEL_ID`; there is no default model. LLM calls fail lazily at first use.
- **`.env` must never be committed** — it holds the LLM API key (ignored via a global gitignore rule; no repo `.gitignore` exists — add one).
- `uv.lock` and `.python-version` are committed in this repo; `evaluation/results/` eval outputs are committed too.
- `data/` and `models/` hold downloaded/generated artifacts — populated by the setup commands above (`data/pokemon.jsonl`, `data/chunks/documents.jsonl`, ONNX embedder under `models/`); the two raw CSVs under `data/raw/` are **bundled and committed** (Kaggle anonymous downloads are bot-blocked, so the repo ships its own copy — no login needed); `evaluation/data/qa.jsonl` is an LLM-generated eval artifact; the span store lives at `monitoring/traces.db`.
- Keep the project self-contained: no imports from external reference
  material (docstring attributions are comments only, never dependencies).

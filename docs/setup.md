# Setup Guide

## Prerequisites

- **Docker + Docker Compose** (recommended) or **Python 3.13+** with [uv](https://docs.astral.sh/uv/)
- **LLM API endpoint** — a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`), configured via `OPENAI_API_BASE_URL` in `.env`
  - Model: `MODEL_ID` is required in `.env` (no default fallback)
  - The LLM API must be reachable before starting the app
- **Tavily API key** (optional) — `TAVILY_API_KEY` in `.env` enables the Bulbapedia web-search fallback when the local knowledge base is insufficient; without it the agent answers from local search only and rejects when it cannot answer confidently

## Development Subset (Default)

All automated runs use the **dev subset**: a deterministic coverage-sampled 50 Pokémon (all 18 types, all generations, legendary/mythical representation) → 50 Pokédex records, 250 ground-truth questions. This is a user directive (2026-08-09): `src/data/build_documents.py` builds the full 1,350-record dataset by default (no LLM cost, seconds), and the dev-subset limit is applied in `evaluation/generate_qa.py` so automated eval runs stay cheap; full-data QA runs are manual only. See [Manual full-data runs](#manual-full-data-runs) below.

## Docker Setup (Recommended)

### 1. Clone and configure

```bash
cd project/
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY="your-api-key-here"
OPENAI_API_BASE_URL="http://localhost:9101/v1"
DATASET_PATH="./data"
TAVILY_API_KEY="your-tavily-api-key-here"   # optional: Bulbapedia web-search fallback
CONFIDENCE_THRESHOLD="0.5"                  # minimum grounding cosine (0..1) for an answer; below it -> rejection
RETRIEVAL_SCORE_THRESHOLD="0.3"             # minimum query↔chunk cosine (0..1) for a retrieved chunk; below it -> dropped
```

### 2. Start services

```bash
docker-compose up --build
```

This starts three services:

| Service    | Container           | Port  | Description                           |
|------------|---------------------|-------|---------------------------------------|
| `app`      | `capstone-app`      | 8501  | Streamlit RAG interface               |
| `postgres` | `capstone-postgres` | 5433  | PostgreSQL (monitoring: spans, conversations, searches, llm_calls, feedback) |
| `grafana`  | `capstone-grafana`  | 3000  | Monitoring dashboard ("Pokemon RAG Monitoring") |

### 3. Wait for pipeline

On first run, the entrypoint script:
1. Seeds the bundled dataset CSVs into the data volume (they ship in the image — no Kaggle login needed; `build_documents.py` falls back to a download only if they are missing)
2. Builds `data/chunks/documents.jsonl` (full dataset: 1,350 Pokémon docs + 18 type-chart docs)
3. Builds hybrid search indices (keyword + vector)
4. Initializes the Postgres monitoring schema (spans, conversations, searches, llm_calls, feedback)
5. Launches Streamlit

Watch the logs for `Pipeline complete! Starting Streamlit...`.

### 4. Access the app

Open `http://localhost:8501` in your browser. Grafana is at `http://localhost:3000` (login `admin` / `admin`).

### 5. Stop services

```bash
docker-compose down
```

Persistent data is stored in Docker volumes (`postgres_data`, `app_data`, `grafana_data`).

To remove all data:

```bash
docker-compose down -v
```

## Local Development Setup

### 1. Install dependencies

```bash
cd project/
uv sync
```

`uv sync` installs all project dependencies from `pyproject.toml` / `uv.lock`.
To add the full dependency set explicitly instead:

```bash
uv add gitsource "huggingface-hub>=1.21" jupyter "matplotlib>=3.10" "minsearch>=0.1" "numpy>=2.5" "onnxruntime>=1.27" "openai>=2.42" "opentelemetry-api>=1.44" "opentelemetry-sdk>=1.44" "psycopg[binary]>=3.3" "python-dotenv>=1.2" "requests>=2.34" "sqlitesearch>=0.1" "streamlit>=1.59" "tavily-python>=0.7" "tokenizers>=0.22" "toyaikit>=0.0.11" "tqdm>=4.68" "watchdog>=6.0" "wget>=3.2"
uv add --group dev pytest          # dev group: test runner (uv sync installs it by default)
```

Package roles: **minsearch** (keyword index), **onnxruntime + tokenizers** (local embeddings), **openai** (LLM Responses API), **tavily-python** (Bulbapedia web search), **streamlit** (chat UI + dashboards), **opentelemetry-*** (tracing), **psycopg** (Postgres), **python-dotenv** (`.env`), **matplotlib** (eval charts), plus data/ingestion helpers (requests, tqdm, watchdog, wget, huggingface-hub, jupyter, gitsource, sqlitesearch, toyaikit, numpy).

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run the pipeline manually

```bash
# Download ONNX embedding model (tokenizer.json + model.onnx)
uv run python -m src.data.download_model

# Download the Pokémon dataset and build documents.jsonl (full dataset: 1,350)
uv run python -m src.data.build_documents

# Generate the ground-truth set (dev subset: 250 questions; requires the LLM API)
uv run python -m evaluation.generate_qa
```

`build_documents.py` defaults to the full 1,350-record dataset; pass `--limit N` for a smaller dataset (e.g. `--limit 50`). `generate_qa.py` defaults to the deterministic coverage-sampled dev subset (50 records × 5 questions = 250) and supports `--full` (all 1,350 records), `--limit N`, `--questions N` (questions per record), `--seed N`, and `--resume` (skip ids already in qa.jsonl). Each row is `{"question", "document"}` — questions only, linked to the Pokédex document that contains the answer; the LLM never writes answers.

### 4. Start the app

```bash
uv run streamlit run src/interface/app.py
```

Open `http://localhost:8501`.

### 5. Run evaluations

Evaluations run as Jupyter notebooks under `evaluation/notebooks/` (builders in `evaluation/notebooks/builders/`; results committed under `evaluation/results/`). Open them with `uv run jupyter notebook evaluation/notebooks` and run top-to-bottom, or execute headless:

```bash
uv run jupyter nbconvert --to notebook --execute evaluation/notebooks/04_retrieval_quality.ipynb   # retrieval (no LLM)
uv run jupyter nbconvert --to notebook --execute evaluation/notebooks/05_answer_quality.ipynb      # LLM judge (~19 min on dev subset)
uv run jupyter nbconvert --to notebook --execute evaluation/notebooks/01_agent_path_analysis.ipynb # agent path analysis (6-question trace)
```

### 6. Open the monitoring dashboard

```bash
# Grafana (needs Postgres spans; Docker Compose starts both)
docker-compose up -d postgres grafana
# then open http://localhost:3000
```

## Manual Full-Data Runs

The full dataset is 1,350 records (1,025 canonical + 325 alternate forms); the full QA set would be 6,750 questions (5 per record). Full runs are manual only — never run by agents, never part of CI.

### Parameter changes

| Step | Command | What it does |
|------|---------|--------------|
| Build documents | `uv run python -m src.data.build_documents` | All 1,350 records + 18 type charts → `data/chunks/documents.jsonl` (default) |
| QA     | `uv run python -m evaluation.generate_qa --full` | 1,350 records × 5 = 6,750 questions (flagged MANUAL — slow/costly) |

`--limit N` on `generate_qa.py` selects a coverage-sampled N records (deterministic, `--seed`); on `build_documents.py` it takes the first N by id.

The evaluation notebooks read the dev-subset `evaluation/data/qa.jsonl` by default. The LLM-judge notebooks (`05_answer_quality.ipynb`, `03_gate_quality_comparison.ipynb`) analyze the gated `evaluation/notebooks/data/gate_collection.jsonl` offline (no sampling).

### Regeneration order

The pipeline is strictly ordered — each step reads the previous step's output:

```bash
uv run python -m src.data.build_documents   # 1. documents.jsonl (1350, default)
uv run python -m evaluation.generate_qa --full   # 2. qa.jsonl (6750) — LLM cost
uv run jupyter nbconvert --to notebook --execute evaluation/notebooks/04_retrieval_quality.ipynb   # 3. retrieval metrics (no LLM)
uv run jupyter nbconvert --to notebook --execute evaluation/notebooks/05_answer_quality.ipynb      # 4. LLM judge (~19 min on dev subset)
uv run jupyter nbconvert --to notebook --execute evaluation/notebooks/01_agent_path_analysis.ipynb # 5. agent path analysis (notebook 01)
```

### Cost / time note

Measured on the dev subset (local qwen via `localhost:9101`): the LLM eval took ≈ 1,120s (~19 min) and the agent eval ≈ 27 min. The full 1,350-Pokémon / 6,750-question run will be substantially longer (build_documents scales linearly, QA generation scales with records, and the evals scale with ground-truth questions) and consumes meaningful LLM tokens — the QA generator alone issues one multi-pair prompt per record. Budget accordingly, and switch back to the dev subset afterwards (a plain `uv run python -m evaluation.generate_qa` run regenerates the coverage-sampled 250-question set).

## Configuration Reference

### Environment Variables

| Variable           | Default                          | Description                          |
|--------------------|----------------------------------|--------------------------------------|
| `OPENAI_API_KEY`   | `your-api-key-here`              | API key for the LLM API (local or cloud) |
| `OPENAI_API_BASE_URL` | (none — required)              | LLM API endpoint — local hosted LLM or cloud OpenAI-compatible API |
| `OPENAI_BASE_URL`     | (alias for `OPENAI_API_BASE_URL`) | Legacy alias for the LLM API endpoint |
| `TRACING_ENABLED`     | `true`                          | When `false`, tracing is disabled and no spans are written to Postgres |
| `MODEL_ID`         | (none — required)                | LLM model name, no default fallback  |
| `DATASET_PATH`     | `./data`                         | Dataset storage path                |
| `TAVILY_API_KEY`   | (none — optional)                | Tavily API key for the Bulbapedia web-search fallback; without it the agent rejects when local search cannot answer confidently |
| `CONFIDENCE_THRESHOLD` | required (set in .env; no default) | Minimum grounding score (0–1) an answer must have to be returned — max embedding-cosine similarity between the answer and any single retrieved record; below it the answer is replaced by the rejection message |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3`                     | Minimum query↔chunk cosine (0–1) a retrieved chunk must have to be returned by hybrid search; below it the chunk is dropped as irrelevant |
| `AGENT_TEMPERATURE` | `0.0`                          | Sampling temperature for agent-loop tool decisions (deterministic) — set to `1` for reasoning models (o1/o3) |
| `EMBEDDER_MODEL_PATH` | (Docker-only)                | Override for the ONNX bi-encoder location (set by `deployment/entrypoint.sh`; not user-facing) |
| `RERANKER_MODEL_PATH` | (Docker-only)                | Override for the ONNX cross-encoder (re-ranker) location (set by `deployment/entrypoint.sh`; not user-facing) |
| `POSTGRES_DB`      | `capstone`                       | PostgreSQL database name            |
| `POSTGRES_USER`    | `capstone`                       | PostgreSQL username                  |
| `POSTGRES_PASSWORD`| `capstone_secret`                | PostgreSQL password                  |
| `POSTGRES_PORT`    | `5432`                           | PostgreSQL host port                |
| `POSTGRES_HOST`    | `localhost`                      | Postgres host for monitoring (spans, conversations, searches, llm_calls, feedback) |
| `APP_PORT`         | `8501`                           | Streamlit app port                  |

### Docker Compose Ports

| Service  | Container Port | Host Port | Protocol |
|----------|----------------|-----------|----------|
| app      | 8501           | 8501      | HTTP     |
| postgres | 5432           | 5433      | TCP      |
| grafana  | 3000           | 3000      | HTTP     |

### Data Flow

```
repo bundle (fallback: Kaggle API) ──► data/raw/pokemon_complete.csv + pokemon_types.csv (1,350 records, bundled)
                    │
                    ▼
              data/chunks/documents.jsonl (1,350 Pokémon docs + 18 type-chart docs, indexed)
                    │
                    ▼
              HybridSearch index (keyword + vector + RRF)
                    │
                    ▼
              RAG Agent (manual tool loop: LLM decides local / Bulbapedia web search)
                    │
                    ▼
              Streamlit UI (chat + feedback)
                    │
                    ▼
              Monitoring (OpenTelemetry → Postgres → Grafana)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

Ensure you're running from the `project/` directory. The `PYTHONPATH` is set to `/app` in Docker, but locally you need to be in the project root.

### "LLM API not available"

The LLM API must be reachable at the configured `OPENAI_API_BASE_URL` (e.g. `http://localhost:9101/v1` for a locally hosted LLM). Check:
```bash
curl http://localhost:9101/v1/models
```

### Docker build fails

Ensure you have enough disk space and memory. The ONNX embedding model download requires ~100MB.

### Streamlit shows "Loading search index and LLM..."

This is normal on first request. The hybrid search index loads ~50 document embeddings into memory on the dev subset. Subsequent requests are fast.

### PostgreSQL connection refused

Check that the postgres container is healthy:
```bash
docker-compose ps
docker-compose logs postgres
```

Monitoring requires Postgres — spans, conversations, and feedback all live in the same database. For local development, run `docker-compose up postgres` (or point `POSTGRES_HOST`/`POSTGRES_PORT` at a local server); the app and dashboard degrade gracefully when Postgres is down, but no monitoring data is recorded until it is reachable.

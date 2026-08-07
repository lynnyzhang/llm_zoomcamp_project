# Setup Guide

## Prerequisites

- **Docker + Docker Compose** (recommended) or **Python 3.13+** with [uv](https://docs.astral.sh/uv/)
- **LLM API endpoint** — a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`), configured via `OPENAI_API_BASE_URL` in `.env`
  - Model: `MODEL_ID` is required in `.env` (no default fallback)
  - The LLM API must be reachable before starting the app

## Development Subset (Default)

All automated runs use the **dev subset**: the first 50 Pokémon by id (50 Pokédex passages, 250 QA pairs). This is a user directive (2026-08-07): the dev subset is the default for every test and evaluation run, and full-data runs are manual only. This keeps CI fast and cheap; see [Manual full-data runs](#manual-full-data-runs) below.

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
```

### 2. Start services

```bash
docker-compose up --build
```

This starts three services:

| Service    | Container           | Port  | Description                           |
|------------|---------------------|-------|---------------------------------------|
| `app`      | `capstone-app`      | 8501  | Streamlit RAG interface               |
| `postgres` | `capstone-postgres` | 5433  | PostgreSQL (persistent span storage)  |
| `grafana`  | `capstone-grafana`  | 3000  | Monitoring dashboard ("Pokemon RAG Monitoring") |

### 3. Wait for pipeline

On first run, the entrypoint script:
1. Downloads the Pokémon dataset from the Kaggle API endpoint
2. Builds `data/corpus.jsonl` (dev subset: 50 passages)
3. Chunks documents and builds hybrid search indices (keyword + vector)
4. Initializes the monitoring SQLite database (and Postgres schema)
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

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run the pipeline manually

```bash
# Download ONNX embedding model (tokenizer.json + model.onnx)
uv run python -m src.data.download_model

# Download the Pokémon dataset and build corpus.jsonl (dev subset: 50)
uv run python -m src.data.ingest

# Chunk documents into data/chunks/documents.jsonl
uv run python -m src.data.chunker

# Generate the QA set (dev subset: 250 pairs; requires the LLM API)
uv run python -m src.evaluation.generate_qa
```

`ingest.py` supports `--limit N` (any N) and `--full` (all 1,025 records); `generate_qa.py` supports the same flags. Defaults always produce the dev subset.

### 4. Start the app

```bash
uv run streamlit run src/interface/app.py
```

Open `http://localhost:8501`.

### 5. Run evaluations

```bash
# Retrieval evaluation (fast, no LLM needed)
uv run python -m src.evaluation.retrieval_eval

# LLM evaluation (requires the LLM API; ~19 min on the dev subset)
uv run python -m src.evaluation.llm_eval

# Agent evaluation (requires the LLM API; ~27 min on the dev subset)
uv run python -m src.evaluation.agent_eval
```

### 6. Open the monitoring dashboard

```bash
# Grafana (needs Postgres spans; Docker Compose starts both)
docker-compose up -d postgres grafana
# then open http://localhost:3000

# Or the Streamlit dashboard (reads the SQLite store directly)
uv run streamlit run src/monitoring/dashboard.py
```

## Manual Full-Data Runs

The full dataset is 1,025 Pokémon; the full QA set would be 5,125 pairs (5 per record). Full runs are manual only — never run by agents, never part of CI.

### Parameter changes

| Step | Command | What it does |
|------|---------|--------------|
| Ingest | `uv run python -m src.data.ingest --full` | All 1,025 records → `data/corpus.jsonl` |
| Chunk  | `uv run python -m src.data.chunker` | Re-chunks whatever corpus exists (no flags) |
| QA     | `uv run python -m src.evaluation.generate_qa --full` | 1,025 records × 5 = 5,125 pairs (flagged MANUAL — slow/costly) |

`--limit N` works the same way on both scripts (first N records by id).

The eval scripts take no CLI flags: `retrieval_eval.py` reads the full `data/qa.jsonl`, while `llm_eval.py` and `agent_eval.py` use in-script `sample_size` constants (10 for the LLM judge, 20 judge / 50 agent-full in `agent_eval.py`). For a full judge pass, edit those constants to `0` (all pairs) before running.

### Regeneration order

The pipeline is strictly ordered — each step reads the previous step's output:

```bash
uv run python -m src.data.ingest --full      # 1. corpus.jsonl (1025)
uv run python -m src.data.chunker            # 2. documents.jsonl
uv run python -m src.evaluation.generate_qa --full   # 3. qa.jsonl (5125) — LLM cost
uv run python -m src.evaluation.retrieval_eval       # 4. retrieval metrics (no LLM)
uv run python -m src.evaluation.llm_eval             # 5. LLM judge (~19 min on dev subset)
uv run python -m src.evaluation.agent_eval           # 6. agent vs simple (~27 min on dev subset)
```

### Cost / time note

Measured on the dev subset (local qwen via `localhost:9101`): the LLM eval took ≈ 1,120s (~19 min) and the agent eval ≈ 27 min. The full 1,025-Pokémon / 5,125-pair run will be substantially longer (ingest + chunk scale linearly, QA generation scales with records, and the evals scale with QA pairs) and consumes meaningful LLM tokens — the QA generator alone issues one multi-pair prompt per record. Budget accordingly, and switch back to the dev subset afterwards (`uv run python -m src.data.ingest` with no flags).

## Configuration Reference

### Environment Variables

| Variable           | Default                          | Description                          |
|--------------------|----------------------------------|--------------------------------------|
| `OPENAI_API_KEY`   | `your-api-key-here`              | API key for the LLM API (local or cloud) |
| `OPENAI_API_BASE_URL` | `http://localhost:9101/v1`    | LLM API endpoint — local hosted LLM or cloud OpenAI-compatible API |
| `MODEL_ID`         | (none — required)                | LLM model name, no default fallback  |
| `DATASET_PATH`     | `./data`                         | Dataset storage path                |
| `POSTGRES_DB`      | `capstone`                       | PostgreSQL database name            |
| `POSTGRES_USER`    | `capstone`                       | PostgreSQL username                  |
| `POSTGRES_PASSWORD`| `capstone_secret`                | PostgreSQL password                  |
| `POSTGRES_PORT`    | `5433`                           | PostgreSQL host port                |
| `POSTGRES_HOST`    | (unset)                          | When set, spans also export to Postgres |
| `APP_PORT`         | `8501`                           | Streamlit app port                  |

### Docker Compose Ports

| Service  | Container Port | Host Port | Protocol |
|----------|----------------|-----------|----------|
| app      | 8501           | 8501      | HTTP     |
| postgres | 5432           | 5433      | TCP      |
| grafana  | 3000           | 3000      | HTTP     |

### Data Flow

```
Kaggle API ──► data/raw/complete_pokedex.json (1,025 records, cached)
                    │
                    ▼
              data/corpus.jsonl (dev: 50 passages)
                    │
                    ▼
              data/chunks/documents.jsonl (chunked, indexed)
                    │
                    ▼
              HybridSearch index (keyword + vector + RRF)
                    │
                    ▼
              RAG Agent (guardrails + iterative search + LLM)
                    │
                    ▼
              Streamlit UI (chat + Pokémon cards + feedback)
                    │
                    ▼
              Monitoring (OpenTelemetry → SQLite + Postgres → Grafana)
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

The app degrades gracefully when Postgres is down — spans fall back to SQLite and nothing crashes.

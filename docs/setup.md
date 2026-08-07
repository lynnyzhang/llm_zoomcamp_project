# Setup Guide

## Prerequisites

- **Docker + Docker Compose** (recommended) or **Python 3.13+** with [uv](https://docs.astral.sh/uv/)
- **LLM API endpoint** — a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`), configured via `OPENAI_API_BASE_URL` in `.env`
  - Model: `MODEL_ID` is required in `.env` (no default fallback)
  - The LLM API must be reachable before starting the app

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

This starts two containers:

| Service    | Container          | Port  | Description                    |
|------------|-------------------|-------|--------------------------------|
| `app`      | `capstone-app`    | 8501  | Streamlit RAG interface        |
| `postgres` | `capstone-postgres`| 5433 | PostgreSQL (persistent storage)|

### 3. Wait for pipeline

On first run, the entrypoint script:
1. Downloads `rag-mini-wikipedia` dataset from HuggingFace
2. Chunks 3,200 passages into document segments
3. Builds hybrid search indices (keyword + vector)
4. Initializes the monitoring SQLite database
5. Launches Streamlit

Watch the logs for `Pipeline complete! Starting Streamlit...`.

### 4. Access the app

Open `http://localhost:8501` in your browser.

### 5. Stop services

```bash
docker-compose down
```

Persistent data is stored in Docker volumes (`postgres_data`, `app_data`).

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

# Download dataset
uv run python -c "from src.data.ingest import main; main()"

# Chunk documents
uv run python -c "from src.data.chunker import main; main()"

# Build search index (verifies everything works)
uv run python -m src.search.hybrid
```

### 4. Start the app

```bash
uv run streamlit run src/interface/app.py
```

Open `http://localhost:8501`.

### 5. Run evaluations

```bash
# Retrieval evaluation (fast, no LLM needed)
uv run python -m src.evaluation.retrieval_eval

# LLM evaluation (requires the LLM API)
uv run python -m src.evaluation.llm_eval

# Agent evaluation (requires the LLM API)
uv run python -m src.evaluation.agent_eval
```

### 6. Open monitoring dashboard

```bash
uv run streamlit run src/monitoring/dashboard.py
```

## Configuration Reference

### Environment Variables

| Variable           | Default                          | Description                          |
|--------------------|----------------------------------|--------------------------------------|
| `OPENAI_API_KEY`   | `your-api-key-here`              | API key for the LLM API (local or cloud) |
| `OPENAI_API_BASE_URL` | `http://localhost:9101/v1`    | LLM API endpoint — local hosted LLM or cloud OpenAI-compatible API |
| `DATASET_PATH`     | `./data`                         | Dataset storage path                |
| `POSTGRES_DB`      | `capstone`                       | PostgreSQL database name            |
| `POSTGRES_USER`    | `capstone`                       | PostgreSQL username                  |
| `POSTGRES_PASSWORD`| `capstone_secret`                | PostgreSQL password                  |
| `POSTGRES_PORT`    | `5433`                           | PostgreSQL host port                |
| `APP_PORT`         | `8501`                           | Streamlit app port                  |

### Docker Compose Ports

| Service    | Container Port | Host Port | Protocol |
|------------|---------------|-----------|----------|
| app        | 8501          | 8501      | HTTP     |
| postgres   | 5432          | 5433      | TCP      |

### Data Flow

```
HuggingFace ──► data/corpus.jsonl (3,200 passages)
                data/qa.jsonl (918 Q&A pairs)
                        │
                        ▼
              data/chunks/documents.jsonl (chunked documents)
                        │
                        ▼
              HybridSearch index (keyword + vector + RRF)
                        │
                        ▼
              RAG Agent (iterative search + LLM)
                        │
                        ▼
              Streamlit UI (chat + agent visualization)
                        │
                        ▼
              Monitoring (OpenTelemetry → SQLite → dashboard)
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

This is normal on first request. The hybrid search index loads ~3,200 document embeddings into memory. Subsequent requests are fast.

### PostgreSQL connection refused

Check that the postgres container is healthy:
```bash
docker-compose ps
docker-compose logs postgres
```

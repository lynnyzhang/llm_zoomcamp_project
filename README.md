# LLM Zoomcamp 2026 Capstone Project

An agentic RAG assistant for Pokémon knowledge, built for the DataTalksClub LLM Zoomcamp 2026 cohort. Ask about a Pokémon's stats, types, weaknesses, evolution line, or abilities and the system retrieves the right Pokédex entries, grounds its answer in them, and shows you the whole process. Combines hybrid search (keyword + vector with Reciprocal Rank Fusion), LLM-driven query reformulation, and a Streamlit chat interface with full agent transparency.

## Project Goal

Given a corpus of Pokédex records (dev subset: coverage-sampled 50 Pokémon, 250 ground-truth questions) from the [Kaggle Pokémon dataset](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types), build a question-answering system that:

1. Retrieves relevant Pokémon entries using hybrid search
2. Uses an agentic loop to reformulate queries when results are insufficient
3. Generates faithful, grounded answers via a local LLM
4. Rejects out-of-scope questions (battle simulation, save files, cheats, non-Pokémon topics)
5. Displays Pokémon cards with official artwork
6. Tracks performance with OpenTelemetry-based monitoring

## Data

The system is built on the **Pokémon Dataset with Stats and Types** from Kaggle ([`patelris/pokemon-dataset-with-stats-and-types`](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types), download endpoint `https://www.kaggle.com/api/v1/datasets/download/patelris/pokemon-dataset-with-stats-and-types`). The two raw CSVs (`pokemon_complete.csv`, `pokemon_types.csv`) ship bundled in `data/raw/` — Kaggle's anonymous download endpoint is bot-blocked, so the repo carries its own copy and no login is needed. `src/data/ingest.py` only attempts a download when they are missing, and builds `data/corpus.jsonl` with one structured record per Pokémon (full: 1,350 — 1,025 canonical + 325 alternate forms).

**Development subset:** `src/data/ingest.py` builds the full 1,350-record corpus by default; `evaluation/generate_qa.py` defaults to a deterministic coverage-sampled dev subset of 50 Pokémon (250 ground-truth questions). This is a user directive: dev subset for all automated test/eval runs, full-data QA runs manual only. See [docs/setup.md](docs/setup.md).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                      │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Chat UI  │  │ Agent Visual │  │ Pokémon Card Grid   │    │
│  └──────────┘  └──────────────┘  └─────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Agent (agent.py)                     │
│                                                             │
│  0. Guardrails: reject out-of-scope queries                 │
│  1. perform_search(query)  ──────────────────┐              │
│  2. analyze_results(query, results) ◄────────┘              │
│  3. if not sufficient: reformulate_query() ──► go to 1      │
│  4. generate_answer(query, all_results)                     │
│                                                             │
│  Max 3 iterations, LLM-driven query reformulation           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Search (hybrid.py)                      │
│                                                             │
│       ┌──────────────┐     ┌──────────────┐                 │
│       │   Keyword    │     │    Vector    │                 │
│       │  (minsearch) │     │ (MiniLM-L6v2)│                 │
│       └──────┬───────┘     └──────┬───────┘                 │
│              │                    │                         │
│              └──────────┬─────────┘                         │
│                         ▼                                   │
│           ┌──────────────────────────┐                      │
│           │  Reciprocal Rank Fusion  │                      │
│           │  score = Σ w/(k + rank)  │                      │
│           └──────────────────────────┘                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Pipeline                                  │
│                                                             │
│ Kaggle API ──► raw CSVs ──► corpus.jsonl                   │
│  (patelris/     (1,350 records,          (one record per    │
│   pokemon-       cached, idempotent)      Pokémon, 1,350)   │
│   dataset)                                                  │
│                         │                                   │
│                         ▼                                   │
│              chunker ──► documents.jsonl (indexed)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Monitoring (OpenTelemetry + SQLite/Postgres)        │
│                                                             │
│  TracerSetup ──► SQLiteSpanExporter ──► traces.db           │
│                 PostgresSpanExporter ──► PostgreSQL         │
│  (spans for         (custom schema)      (queries, tokens,  │
│   agent.run,                             feedback, latency) │
│   search, LLM)                                              │
│                                                             │
│  Dashboard: Grafana at http://localhost:3000                │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and OPENAI_API_BASE_URL to your LLM API endpoint (local hosted LLM or cloud OpenAI-compatible API)
docker-compose up --build
```

The app runs at `http://localhost:8501`, Postgres on port 5433, and Grafana at `http://localhost:3000`. The entrypoint ingests the full corpus, chunks it, and starts the app.

### Local Development

```bash
uv sync
cp .env.example .env              # then edit the LLM vars
uv run python -m src.data.download_model   # fetches ONNX embedder (tokenizer.json + model.onnx)
uv run python -m src.data.ingest           # downloads Kaggle dataset, builds data/corpus.jsonl (full dataset, 1,350)
uv run python -m src.data.chunker          # builds data/chunks/documents.jsonl
uv run streamlit run src/interface/app.py  # chat UI at :8501
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

## Usage

```bash
# Ask a question via the Streamlit UI at http://localhost:8501

# Or run the agent from CLI:
set -a; source .env; set +a; uv run python -c "from src.rag.agent import RAGAgent; a = RAGAgent(); r = a.run('What are Pikachu's stats?'); print(r['answer'][:200])"

# Generate the ground-truth set (dev subset, 250 questions):
uv run python -m evaluation.generate_qa

# Run evaluations:
uv run python -m evaluation.retrieval_eval
uv run python -m evaluation.llm_eval
uv run python -m evaluation.agent_eval

# Open the monitoring dashboard:
docker-compose up -d grafana   # then open http://localhost:3000
```

See [docs/usage.md](docs/usage.md) for the complete usage guide.

## Capabilities

- **Hybrid search** — keyword (minsearch) + vector (local ONNX MiniLM) fused with Reciprocal Rank Fusion, selectable as hybrid / keyword / vector in the sidebar.
- **Agentic loop** — up to 3 iterations of LLM-driven query reformulation when the first search is insufficient.
- **Guardrails** — out-of-scope rejection (battle simulation/prediction, save files, cheats, non-Pokémon topics) with a deterministic fail-safe; in-domain low-confidence questions get a graceful "couldn't find a confident answer" response instead.
- **Pokémon cards** — retrieved documents render as cards with official artwork (PokeAPI sprites), types, and a stats excerpt.
- **Feedback capture** — thumbs up/down per answer, recorded in monitoring for continuous improvement.

## Evaluation Results

Measured on the Pokémon dev subset (50 docs, 250 ground-truth questions) with a local qwen model. Full details in [docs/evaluation.md](docs/evaluation.md).

### Retrieval (250 questions, top-5)

| Method    | Precision@5 | Recall@5 | MRR    | Time   |
|-----------|-------------|----------|--------|--------|
| Keyword   | 0.1952      | 0.9760   | 0.9548 | 0.1s   |
| Vector    | 0.1968      | 0.9840   | 0.8910 | 0.4s   |
| **Hybrid**| **0.1968**  | **0.9840**| **0.9520**| **0.6s** |

### LLM Answer Quality (10-question sample, 1-5 scale)

| Prompt       | Faithfulness | Relevance | Coherence |
|--------------|-------------|-----------|-----------|
| Simple       | 3.4         | 4.4       | 4.9       |
| Detailed     | 3.0         | 4.6       | 4.8       |
| **With Examples** | **3.9** | **4.7** | **4.9** |

The with-examples judge prompt scores best overall.

### Agent vs Simple RAG

| Metric                      | Simple RAG | Agentic RAG | Delta      |
|-----------------------------|------------|-------------|------------|
| Retrieval Hit Rate          | 98.4% (246/250) | 98.0% (49/50) | -0.4%   |
| Avg Searches/Query          | 1.0        | 1.22        | +0.22      |
| Latency/Query (LLM)         | 19.26s     | 39.99s      | +20.73s    |
| Answer Quality (LLM Judge)  | 3.9/5      | 3.65/5      | -0.25      |

Both pipelines sit at a ~98% retrieval ceiling on the 50-doc dev subset, so the agent's reformulation adds latency without a hit-rate gain here; it is expected to help on the full corpus where single-shot retrieval is weaker.

## Monitoring

Tracing runs through OpenTelemetry. Every agent run, search, and LLM call produces spans with query, tokens, latency, and feedback:

- **SQLite** (`monitoring/traces.db`) — always on, default store.
- **PostgreSQL** — optional span export when `POSTGRES_HOST` is set (Docker Compose starts Postgres by default).
- **Grafana** at `http://localhost:3000` — dashboard "Pokemon RAG Monitoring" with 10 panels: total traces, cost, average latency, token usage, queries over time, feedback distribution, latency and token trends, top queries, and agent iteration distribution.

## Project Structure

```
project/
├── README.md
├── docker-compose.yml      # Docker orchestration (app + Postgres + Grafana)
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
├── data/
│   ├── raw/pokemon_complete.csv + pokemon_types.csv  # Cached raw CSVs
│   ├── corpus.jsonl        # One structured record per Pokémon (full: 1,350; --limit for subset)
│   └── chunks/documents.jsonl     # 1,350 Pokémon docs + 18 type-chart docs (indexed)
├── models/
│   └── Xenova/all-MiniLM-L6-v2/   # ONNX embedder (tokenizer.json + model.onnx)
├── monitoring/
│   ├── tracer.py           # OpenTelemetry tracing (SQLite + Postgres exporters)
│   ├── dashboard.py        # Streamlit monitoring dashboard
│   ├── traces.db           # Monitoring data (SQLite)
│   ├── grafana/provisioning/   # Grafana datasource + dashboard provisioning
│   └── dashboards/
│       └── pokemon_rag.json    # Grafana dashboard "Pokemon RAG Monitoring"
├── deployment/
│   ├── Dockerfile          # App container (Python 3.13 + uv)
│   ├── .dockerignore       # Build exclusions (context: repo root)
│   └── entrypoint.sh       # Pipeline orchestration
├── evaluation/
│   ├── evaluation_utils.py  # Shared evaluation helpers
│   ├── generate_qa.py       # LLM-generated ground-truth questions
│   ├── retrieval_eval.py    # Retrieval evaluation
│   ├── llm_eval.py          # LLM evaluation
│   ├── agent_eval.py        # Agent evaluation
│   ├── data/
│   │   └── qa.jsonl         # question → document (dev: 250)
│   └── results/
│       ├── retrieval_eval.json
│       ├── llm_eval.json
│       ├── agent_eval.json
│       ├── agent_eval_comparison.png
│       └── final_report.md     # Final evaluation report
├── src/
│   ├── data/
│   │   ├── ingest.py       # Kaggle dataset download → corpus.jsonl
│   │   ├── chunker.py      # Pokémon-native docs (1 per Pokémon + 18 type-chart docs)
│   │   └── download_model.py   # ONNX embedding model download
│   ├── search/
│   │   ├── embedder.py     # ONNX embedder (onnxruntime, no torch)
│   │   └── hybrid.py       # Hybrid search (keyword + vector + RRF)
│   ├── rag/
│   │   ├── pipeline.py     # Base RAG pipeline
│   │   └── agent.py        # Agentic RAG + guardrails + reformulation
│   └── interface/
│       └── app.py          # Streamlit chat UI
├── tests/                  # Test suite (116 tests)
└── notebooks/              # Exploration notebooks
```

## Key Dependencies

- **minsearch** — keyword search index (NOT vector search despite the name)
- **ONNX + tokenizers** — local embeddings (all-MiniLM-L6-v2 via onnxruntime, no torch)
- **OpenAI** — LLM calls via the Responses API to the configured endpoint (locally hosted LLM or cloud OpenAI-compatible API)
- **Streamlit** — chat interface and monitoring dashboard
- **OpenTelemetry** — distributed tracing with SQLite/Postgres storage
- **Grafana** — monitoring dashboards on top of Postgres

## LLM Backend

LLM calls go through the OpenAI Responses API to the endpoint configured via `OPENAI_API_BASE_URL` in `.env` — either a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`). The endpoint must be reachable for any LLM call to work. Model: set `MODEL_ID` in `.env` (required, no default). In Docker, a `localhost` base URL is automatically rewritten to `host.docker.internal` so the container can reach a locally hosted LLM on the host; cloud URLs pass through unchanged.

All LLM calls use the OpenAI Responses API (`client.responses.create`), not Chat Completions.

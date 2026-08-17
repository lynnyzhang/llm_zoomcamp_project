# LLM Zoomcamp 2026 Capstone Project

An agentic RAG assistant for Pokémon knowledge, built for the DataTalksClub LLM Zoomcamp 2026 cohort. Ask about a Pokémon's stats, types, weaknesses, evolution line, or abilities and the system retrieves the right Pokédex entries, grounds its answer in them, and shows you the whole process. Combines hybrid search (keyword + vector with Reciprocal Rank Fusion), LLM-driven tool use (local + Bulbapedia search), and a Streamlit chat interface with full agent transparency.

## Project Goal

Given a dataset of Pokédex records (dev subset: coverage-sampled 50 Pokémon, 250 ground-truth questions) from the [Kaggle Pokémon dataset](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types), build a question-answering system that:

1. Retrieves relevant Pokémon entries using hybrid search
2. Uses an agentic loop to reformulate queries when results are insufficient
3. Generates faithful, grounded answers via a local LLM
4. Rejects out-of-scope questions (battle simulation, save files, cheats, non-Pokémon topics)
5. Displays Pokémon cards with official artwork
6. Tracks performance with OpenTelemetry-based monitoring

## Data

The system is built on the **Pokémon Dataset with Stats and Types** from Kaggle ([`patelris/pokemon-dataset-with-stats-and-types`](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types), download endpoint `https://www.kaggle.com/api/v1/datasets/download/patelris/pokemon-dataset-with-stats-and-types`). The two raw CSVs (`pokemon_complete.csv`, `pokemon_types.csv`) ship bundled in `data/raw/` — Kaggle's anonymous download endpoint is bot-blocked, so the repo carries its own copy and no login is needed. `src/data/build_documents.py` only attempts a download when they are missing, and builds `data/chunks/documents.jsonl` (full: 1,350 Pokémon docs + 18 type-chart docs — 1,025 canonical + 325 alternate forms).

**Development subset:** `src/data/build_documents.py` builds the full 1,350-record dataset by default; `evaluation/generate_qa.py` defaults to a deterministic coverage-sampled dev subset of 50 Pokémon (250 ground-truth questions). This is a user directive: dev subset for all automated test/eval runs, full-data QA runs manual only. See [docs/setup.md](docs/setup.md).

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
│  3. if insufficient: call search_bulbapedia() ──► go to 1   │
│  4. generate_answer(query, all_results)                     │
│                                                             │
│  Max 3 iterations, LLM decides when to call the tools       │
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
│ Kaggle API ──► raw CSVs ──► documents.jsonl               │
│  (patelris/     (1,350 records,          (1,350 Pokémon    │
│   pokemon-       cached, idempotent)      + 18 type charts)│
│   dataset)                                                  │
│                         │                                   │
│                         ▼                                   │
│              HybridSearch (keyword + vector + RRF)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Monitoring (OpenTelemetry + Postgres)            │
│                                                             │
│  TracerSetup ──► PostgresSpanExporter ──► PostgreSQL        │
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

The app runs at `http://localhost:8501`, Postgres on port 5433, and Grafana at `http://localhost:3000`. The entrypoint builds the documents corpus, indexes it, and starts the app.

### Local Development

```bash
uv sync                          # installs ALL project dependencies into .venv (see pyproject.toml)
cp .env.example .env              # then edit the LLM vars
uv run python -m src.data.download_model   # fetches ONNX embedder (tokenizer.json + model.onnx)
uv run python -m src.data.build_documents   # builds data/chunks/documents.jsonl (full: 1,350; --limit for subset)
uv run streamlit run src/interface/app.py  # chat UI at :8501
```

**All project dependencies** (installed by `uv sync`; alternatively add them explicitly):

```bash
uv add gitsource "huggingface-hub>=1.21" jupyter "matplotlib>=3.10" "minsearch>=0.1" "numpy>=2.5" "onnxruntime>=1.27" "openai>=2.42" "opentelemetry-api>=1.44" "opentelemetry-sdk>=1.44" "psycopg[binary]>=3.3" "python-dotenv>=1.2" "requests>=2.34" "sqlitesearch>=0.1" "streamlit>=1.59" "tavily-python>=0.7" "tokenizers>=0.22" "toyaikit>=0.0.11" "tqdm>=4.68" "watchdog>=6.0" "wget>=3.2"
uv add --group dev pytest          # dev group: test runner (uv sync installs it by default)
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

## Usage

```bash
# Ask a question via the Streamlit UI at http://localhost:8501

# Or run the agent from CLI:
set -a; source .env; set +a; uv run python -c "from src.rag.rag_agent import RAGAgent; from src.search.hybrid_search import HybridSearch; a = RAGAgent(search_index=HybridSearch()); r = a.run('What are Pikachu's stats?'); print(r['answer'][:200])"

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

- **Hybrid search** — keyword (minsearch) + vector (local ONNX MiniLM) fused with Reciprocal Rank Fusion.
- **Web search (Bulbapedia)** — Tavily-backed web search for facts the local knowledge base lacks (moves, anime, manga, lore, game history, strategy). The agent decides when local results are insufficient and searches Bulbapedia; if it answers without grounding, the loop forces one Bulbapedia retry before rejecting.
- **Agentic loop** — the LLM decides when to call its tools (`search_local_knowledge_base`, `search_bulbapedia`), up to 3 tool-use rounds, and writes its own web keyword queries (recorded per search as `search_query`).
- **Guardrails** — out-of-scope rejection (battle simulation/prediction, save files, cheats, non-Pokémon topics); the model refuses without calling tools, and the loop never fabricates an answer when searches fail.
- **Pokémon cards** — retrieved documents render as cards with official artwork (PokeAPI sprites), types, and a stats summary.
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

Both pipelines sit at a ~98% retrieval ceiling on the 50-doc dev subset, so the agent loop adds latency without a hit-rate gain here; it is expected to help on the full dataset where single-shot retrieval is weaker.

### Evaluation criteria coverage

Mapped against the course project rubric (see `project.md`):

- **Retrieval evaluation** — three approaches (keyword / vector / hybrid) evaluated on the 250-question dev set; the best (hybrid) is what production uses.
- **LLM evaluation** — multiple approaches compared: three prompt styles (Simple / Detailed / With Examples, LLM-judged) and Simple RAG vs Agentic RAG; the winner of each comparison is the production choice.
- **Best practices** — hybrid search combining text + vector, evaluated; document re-ranking with a cross-encoder over the fused top-N results; user query rewriting via the agent's per-tool keyword queries (recorded as `search_query` per search).
- **Monitoring** — user feedback (thumbs up/down) + Grafana dashboards (10+ charts).
- **Config sweeps** — `uv run python -m evaluation.config_sweep --knob temperature --values 0.0,0.2` (or `--knob confidence_threshold`) runs the agent eval per setting and saves side-by-side results, so the `.env` defaults are chosen with data.

### Screenshots

| App | Where |
|---|---|
| Chat UI with a grounded answer (cards, source caption, feedback) | `docs/screenshots/ui.png` |
| Monitoring dashboard (conversations, traces, feedback) | `docs/screenshots/dashboard.png` |
| Grafana "Pokemon RAG Monitoring" | `docs/screenshots/grafana.png` |

## Monitoring

Tracing runs through OpenTelemetry. Every agent run, search, and LLM call produces spans with query, tokens, latency, and feedback. All monitoring data (conversations, spans, searches, llm_calls, feedback) lives in **PostgreSQL** (Docker Compose starts Postgres by default; `POSTGRES_*` env vars configure it):

- **Grafana** at `http://localhost:3000` — file-provisioned dashboards "Pokemon RAG Monitoring" (10 panels: total traces, cost, average latency, token usage, queries over time, feedback distribution, latency and token trends, top queries, agent iteration distribution) and "Pokemon RAG History" (gated-query rate, answer-path mix, recent conversations, thumbs up/down).

## Documentation

- [docs/setup.md](docs/setup.md) — environment setup, dataset ingestion, index building, and configuration (including every env variable)
- [docs/usage.md](docs/usage.md) — how to use the app, the agent, tracing/monitoring, and the dashboards
- [docs/evaluation.md](docs/evaluation.md) — offline evaluation: QA generation, retrieval/LLM/agent evals, and config sweeps

## Project Structure

```
project/
├── README.md
├── docker-compose.yml      # Docker orchestration (app + Postgres + Grafana)
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
├── data/
│   ├── raw/pokemon_complete.csv + pokemon_types.csv  # Cached raw CSVs
│   └── chunks/documents.jsonl     # 1,350 Pokémon docs + 18 type-chart docs (indexed)
├── models/
│   └── Xenova/all-MiniLM-L6-v2/   # ONNX embedder (tokenizer.json + model.onnx)
├── monitoring/
│   ├── tracer.py           # OpenTelemetry tracing (Postgres span store)
│   ├── grafana/provisioning/   # Grafana datasource + dashboard provisioning
│   └── dashboards/
│       ├── pokemon_rag.json        # Grafana dashboard "Pokemon RAG Monitoring"
│       └── pokemon_rag_history.json # Grafana dashboard "Pokemon RAG History"
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
│   │   ├── build_documents.py   # raw CSVs → documents.jsonl (corpus entry point)
│   │   └── download_model.py   # ONNX embedding model download
│   ├── search/
│   │   ├── embedder.py     # ONNX embedder (onnxruntime, no torch)
│   │   ├── hybrid_search.py # Hybrid search (keyword + vector + RRF)
│   │   ├── reranker.py      # Cross-encoder re-ranking over fused results
│   │   └── web_search.py   # Bulbapedia web search (Tavily)
│   ├── rag/
│   │   ├── rag_base.py      # Base RAG pipeline
│   │   └── rag_agent.py     # Agentic RAG: LLM tool calls + guardrails
│   └── interface/
│       ├── app.py          # Streamlit chat entry
│       ├── chat_page.py    # Agent-loop orchestration
│       ├── message_renderer.py  # Chat bubble rendering
│       └── card_renderer.py     # Pokémon cards
├── tests/                  # Test suite (142 tests)
└── notebooks/              # Exploration notebooks
```

## Key Dependencies

- **minsearch** — keyword search index (NOT vector search despite the name)
- **ONNX + tokenizers** — local embeddings (all-MiniLM-L6-v2 via onnxruntime, no torch)
- **Tavily** — Bulbapedia web search backend for the agent's escalation path (`TAVILY_API_KEY`; a missing key degrades gracefully to empty results)
- **OpenAI** — LLM calls via the Responses API to the configured endpoint (locally hosted LLM or cloud OpenAI-compatible API)
- **Streamlit** — chat interface and monitoring dashboard
- **OpenTelemetry** — distributed tracing with Postgres storage
- **Grafana** — monitoring dashboards on top of Postgres

## LLM Backend

LLM calls go through the OpenAI Responses API to the endpoint configured via `OPENAI_API_BASE_URL` in `.env` — either a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`). The endpoint must be reachable for any LLM call to work. Model: set `MODEL_ID` in `.env` (required, no default). In Docker, a `localhost` base URL is automatically rewritten to `host.docker.internal` so the container can reach a locally hosted LLM on the host; cloud URLs pass through unchanged.

All LLM calls use the OpenAI Responses API (`client.responses.create`), not Chat Completions.

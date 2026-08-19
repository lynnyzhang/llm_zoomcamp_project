# Agentic Pokemon Knowledge RAG

An agentic RAG assistant for Pokémon knowledge, built for the DataTalksClub LLM Zoomcamp 2026 cohort. Ask about a Pokémon's stats, types, weaknesses, evolution line, or abilities and the system retrieves the right Pokédex entries, grounds its answer in them, and shows you the whole process. Combines hybrid search (keyword + vector with Reciprocal Rank Fusion), cross-encoder re-ranking, LLM-driven tool use (local + Bulbapedia search), and a Streamlit chat interface with full agent transparency.

## Project Goal

Given a dataset of Pokédex records (dev subset: coverage-sampled 50 Pokémon, 250 ground-truth questions) from the [Kaggle Pokémon dataset](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types), build a question-answering system that:

1. Retrieves relevant Pokémon entries using hybrid search
2. Uses an agentic loop to reformulate queries when results are insufficient
3. Generates faithful, grounded answers via a local LLM
4. Rejects out-of-scope questions (battle simulation, save files, cheats, non-Pokémon topics)
5. Tracks performance with OpenTelemetry-based monitoring

## Data

The system is built on the **Pokémon Dataset with Stats and Types** from Kaggle ([`patelris/pokemon-dataset-with-stats-and-types`](https://www.kaggle.com/datasets/patelris/pokemon-dataset-with-stats-and-types)). The two raw CSVs (`pokemon_complete.csv`, `pokemon_types.csv`) ship **bundled** in `data/raw/` — Kaggle's anonymous download endpoint is bot-blocked, so the repo carries its own copy and no login is needed. `src/data/build_documents.py` only attempts a download when they are missing, and builds `data/chunks/documents.jsonl`: each Pokémon's `search_text` is split into **token-aware chunks** (100-token windows, 50-token step, each ≤128 tokens so it fits the embedding window), giving **6,082 Pokémon chunks + 18 type-chart docs** (1,025 canonical + 325 alternate forms).

**Development subset:** `src/data/build_documents.py` builds the full 1,350-record dataset by default; `evaluation/data/src/generate_qa.py` defaults to a deterministic coverage-sampled dev subset of 50 Pokémon (250 ground-truth questions). This is a user directive: dev subset for all automated test/eval runs, full-data QA runs manual only. See [docs/setup.md](docs/setup.md).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                      │
│  ┌──────────┐  ┌──────────────┐                             │
│  │ Chat UI  │  │ Agent Visual │                             │
│  └──────────┘  └──────────────┘                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Agent (rag_agent.py)                 │
│                                                             │
│  0. Guardrails: reject out-of-scope queries                 │
│  1. perform_search(query)  ──────────────────┐              │
│  2. analyze_results(query, results) ◄────────┘              │
│  3. if insufficient: call search_bulbapedia() ──► go to 1   │
│  4. generate_answer(query, all_results)                     │
│                                                             │
│  Max 5 iterations, LLM decides when to call the tools       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Search (hybrid_search.py)               │
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
│           └────────────┬─────────────┘                      │
│                        ▼                                   │
│           ┌──────────────────────────┐                      │
│           │  Cross-encoder reranker  │                      │
│           │  (ms-marco-MiniLM-L-6-v2)│                      │
│           └──────────────────────────┘                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Pipeline                                  │
│                                                             │
│ Bundled raw CSVs ──► documents.jsonl                       │
│  (data/raw/,         (1,350 Pokémon → 6,082                │
│   shipped in repo)    token-aware chunks                    │
│                       + 18 type charts)                     │
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
uv run python -m src.data.build_documents   # builds data/chunks/documents.jsonl (full: 1,350 Pokémon → 6,082 chunks; --limit for subset)
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
set -a; source .env; set +a; uv run python -c "from src.rag.rag_agent import RAGAgent; from src.search.hybrid_search import HybridSearch; a = RAGAgent(search_index=HybridSearch()); r = a.run('What are Pikachu's stats?'); print(r.answer[:200])"

# Generate the ground-truth set (dev subset, 250 questions):
uv run python -m evaluation.data.src.generate_qa

# Run evaluations via the notebooks under evaluation/notebooks/:
#   01_agent_path_analysis, 02_gate_calibration, 03_gate_quality_comparison,
#   04_retrieval_quality, 05_answer_quality
# (see docs/evaluation.md)

# Open the monitoring dashboard:
docker-compose up -d grafana   # then open http://localhost:3000
```

See [docs/usage.md](docs/usage.md) for the complete usage guide.

## Capabilities

- **Hybrid search** — keyword (minsearch) + vector (local ONNX MiniLM) fused with Reciprocal Rank Fusion, then re-ranked by a cross-encoder (ms-marco-MiniLM-L-6-v2).
- **Web search (Bulbapedia)** — Tavily-backed web search for facts the local knowledge base lacks (moves, anime, manga, lore, game history, strategy). The agent decides when local results are insufficient and searches Bulbapedia; if it answers without grounding, the loop forces one Bulbapedia retry before rejecting.
- **Agentic loop** — the LLM decides when to call its tools (`search_local_knowledge_base`, `search_bulbapedia`), up to 5 tool-use rounds, and writes its own web keyword queries (recorded per search as `search_query`). For multi-Pokémon questions it searches once per Pokémon.
- **Guardrails** — out-of-scope rejection (battle simulation/prediction, save files, cheats, non-Pokémon topics); the model refuses without calling tools, and the loop never fabricates an answer when searches fail.
- **Feedback capture** — thumbs up/down per answer, recorded in monitoring for continuous improvement.

## Evaluation Results

Evaluation runs via the notebooks under `evaluation/notebooks/` on the chunked corpus. Full details in [docs/evaluation.md](docs/evaluation.md).

### Retrieval quality (notebook 04 — 250 questions, top-5, chunked corpus)

| Method    | Hit@5 | Precision@5 | Recall@5 | MRR    |
|-----------|-------|-------------|----------|--------|
| Keyword   | 0.856 | 0.171       | 0.856    | 0.814  |
| Vector    | 0.704 | 0.141       | 0.704    | 0.595  |
| **Hybrid**| **0.856** | **0.171** | **0.856** | **0.829** |

Hybrid matches keyword recall (0.856) and beats vector on every metric. The keyword half preserves recall for tail-line facts (evolution, type effectiveness) that the 128-token vector embedding truncates.

### Answer quality (notebook 05 — 49 questions, 29 accepted, 1-5 scale)

| Dimension    | Mean (accepted) |
|--------------|-----------------|
| Faithfulness | 3.966           |
| Relevance    | 4.586           |
| Coherence    | 4.690           |

Per-type (n / acceptance / faithfulness / relevance / coherence):

| Type      | n  | Acceptance | Faithfulness | Relevance | Coherence |
|-----------|----|-----------|--------------|-----------|-----------|
| ability   | 9  | 0.333     | 5.000        | 5.000     | 5.000     |
| evolution | 7  | 0.429     | 5.000        | 5.000     | 5.000     |
| other     | 8  | 0.750     | 3.667        | 4.667     | 4.667     |
| stats     | 14 | 0.500     | 3.429        | 4.000     | 4.429     |
| type      | 11 | 0.909     | 3.900        | 4.700     | 4.700     |

`stats` questions are the main quality/grounding bottleneck (lowest acceptance and faithfulness); `evolution` answers cleanly.

### Evaluation criteria coverage

Mapped against the course project rubric (see `project.md`):

- **Retrieval evaluation** — notebook 04 compares keyword / vector / hybrid on the 250-question dev set; the best (hybrid) is what production uses.
- **LLM evaluation** — notebook 05 measures answer quality (faithfulness / relevance / coherence) per question type via an LLM judge on the accepted answers.
- **Best practices** — hybrid search combining text + vector, evaluated; cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) over the fused top-N results; user query rewriting via the agent's per-tool keyword queries (recorded as `search_query` per search).
- **Monitoring** — user feedback (thumbs up/down) + Grafana dashboards.
- **Config tuning** — gate thresholds and retrieval settings are tuned with data via the notebooks (02 gate calibration, 03 gate-quality comparison) rather than a standalone sweep script.

### Screenshots

| App | Where |
|---|---|
| Chat UI with a grounded answer (cards, source caption, feedback) | `docs/screenshots/ui.png` |
| Grafana "Pokemon RAG Monitoring" | `docs/screenshots/grafana.png` |

## Monitoring

Tracing runs through OpenTelemetry. Every agent run, search, and LLM call produces spans with query, tokens, latency, and feedback. All monitoring data (conversations, spans, searches, llm_calls, feedback) lives in **PostgreSQL** (Docker Compose starts Postgres by default; `POSTGRES_*` env vars configure it):

- **Grafana** at `http://localhost:3000` — three file-provisioned dashboards:
  - **Pokemon RAG Monitoring** (6 panels): total traces, average latency, total tokens, feedback distribution, top queries, agent iterations distribution.
  - **Pokemon RAG History** (10 panels): total turns, gated turns, gated rate, error turns, gated-query rate over time, answer-path mix, conversations over time, recent conversations, thumbs up/down, negative feedback with trace.
  - **Pokemon RAG Spans** (5 panels): recent agent runs, search count distribution, search queries, agent iterations, token usage per query.

## Documentation

- [docs/setup.md](docs/setup.md) — environment setup, dataset ingestion, index building, and configuration (including every env variable)
- [docs/usage.md](docs/usage.md) — how to use the app, the agent, tracing/monitoring, and the dashboards
- [docs/evaluation.md](docs/evaluation.md) — offline evaluation: QA generation, retrieval/answer-quality evals, and gate calibration via the notebooks

## Project Structure

```
project/
├── README.md
├── docker-compose.yml      # Docker orchestration (app + Postgres + Grafana)
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
├── data/
│   ├── raw/pokemon_complete.csv + pokemon_types.csv  # Bundled raw CSVs
│   └── chunks/documents.jsonl     # 6,082 Pokémon chunks + 18 type-chart docs (indexed)
├── models/
│   ├── Xenova/all-MiniLM-L6-v2/   # ONNX embedder (tokenizer.json + model.onnx)
│   └── Xenova/ms-marco-MiniLM-L-6-v2/  # Cross-encoder reranker (model.onnx)
├── monitoring/
│   ├── tracer.py           # OpenTelemetry tracing (Postgres span store)
│   ├── span_exporter.py    # Postgres span exporter
│   ├── span_store.py       # Span store + feedback/stats queries
│   ├── db_init.py          # Postgres connection + schema
│   ├── db_save.py          # Save conversations, searches, llm_calls
│   ├── db_feedback.py      # Save user feedback
│   ├── db_query.py          # Query helpers
│   ├── db_stats.py          # Aggregated stats
│   ├── grafana/provisioning/   # Grafana datasource + dashboard provisioning
│   └── dashboards/
│       ├── pokemon_rag.json        # Grafana dashboard "Pokemon RAG Monitoring" (6 panels)
│       ├── pokemon_rag_history.json # Grafana dashboard "Pokemon RAG History" (10 panels)
│       └── pokemon_rag_spans.json   # Grafana dashboard "Pokemon RAG Spans" (5 panels)
├── deployment/
│   ├── Dockerfile          # App container (Python 3.13 + uv)
│   ├── .dockerignore       # Build exclusions (context: repo root)
│   └── entrypoint.sh       # Pipeline orchestration
├── evaluation/
│   ├── data/
│   │   ├── qa.jsonl              # question → document (dev: 250)
│   │   ├── gate_collection.jsonl # judged gate data (notebooks 02/03/05 input)
│   │   └── src/                  # QA + gate-data generation scripts
│   │       ├── generate_qa.py    # LLM-generated ground-truth questions
│   │       ├── qa_generation.py  # question generation + dev-subset sampling
│   │       └── collect_gate_data.py # judged gate-data collection
│   └── notebooks/                # Evaluation notebooks (01-05)
│       ├── 01_agent_path_analysis.ipynb
│       ├── 02_gate_calibration.ipynb
│       ├── 03_gate_quality_comparison.ipynb
│       ├── 04_retrieval_quality.ipynb
│       ├── 05_answer_quality.ipynb
│       ├── share/                # Shared eval code (imported by notebooks + scripts)
│       │   ├── common.py         # agent build, qa loading, tracing
│       │   ├── document_index.py # ground-truth doc reconstruction from chunks
│       │   ├── judge_prompts.py  # judge prompt variants
│       │   ├── llm_calls.py      # structured LLM calls (Responses API)
│       │   └── judge.py          # correctness + grounding judges
│       └── results/              # committed notebook outputs
│           └── agent_path_trace.txt
├── src/
│   ├── llm/
│   │   └── env.py           # LLM env config (API key, base URL, model id)
│   ├── data/
│   │   ├── build_documents.py   # raw CSVs → chunked documents.jsonl (corpus entry point)
│   │   ├── chunking.py          # Token-aware sliding-window chunking
│   │   ├── csv_parsers.py       # Raw CSV parsing
│   │   ├── download.py          # Optional CSV download
│   │   ├── download_model.py   # ONNX embedding model download
│   │   ├── evolution.py         # Evolution-chain building
│   │   ├── evolution_overrides.py # Evolution overrides
│   │   ├── pokemon_doc_builder.py # Pokémon document assembly
│   │   └── type_chart.py        # Type-chart documents
│   ├── search/
│   │   ├── embedder.py     # ONNX embedder (onnxruntime, no torch)
│   │   ├── hybrid_search.py # Hybrid search (keyword + vector + RRF + relevance filter)
│   │   ├── reranker.py      # Cross-encoder re-ranking over fused results
│   │   ├── search_records.py # Search result / record dataclasses
│   │   └── web_search.py   # Bulbapedia web search (Tavily)
│   ├── rag/
│   │   ├── rag_base.py      # Base RAG pipeline
│   │   ├── rag_agent.py     # Agentic RAG: LLM tool calls + guardrails
│   │   ├── tools.py         # Tool definitions + SearchRecord
│   │   ├── prompts.py       # Agent/answer prompts
│   │   ├── scoring.py       # Line-level grounding gate
│   │   └── llm_call_record.py # LLM call record + cost
│   └── interface/
│       ├── app.py          # Streamlit chat entry
│       ├── chat_page.py    # Agent-loop orchestration
│       ├── chat_message.py # Chat message model
│       ├── message_renderer.py  # Chat bubble rendering
│       └── agent_loop_saver.py # Persists agent loop + feedback to monitoring
└── tests/                  # Test suite (124 tests)
```

## Key Dependencies

- **minsearch** — keyword search index (NOT vector search despite the name)
- **ONNX + tokenizers** — local embeddings (all-MiniLM-L6-v2 via onnxruntime, no torch) and cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- **Tavily** — Bulbapedia web search backend for the agent's escalation path (`TAVILY_API_KEY`; a missing key degrades gracefully to empty results)
- **OpenAI** — LLM calls via the Responses API to the configured endpoint (locally hosted LLM or cloud OpenAI-compatible API)
- **Streamlit** — chat interface
- **OpenTelemetry** — distributed tracing with Postgres storage
- **Grafana** — monitoring dashboards on top of Postgres

## LLM Backend

LLM calls go through the OpenAI Responses API to the endpoint configured via `OPENAI_API_BASE_URL` in `.env` — either a locally hosted LLM (e.g. `http://localhost/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`). The endpoint must be reachable for any LLM call to work. Model: set `MODEL_ID` in `.env` (required, no default). In Docker, a `localhost` base URL is automatically rewritten to `host.docker.internal` so the container can reach a locally hosted LLM on the host; cloud URLs pass through unchanged.

All LLM calls use the OpenAI Responses API (`client.responses.create`), not Chat Completions.

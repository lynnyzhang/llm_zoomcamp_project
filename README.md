# LLM Zoomcamp 2026 Capstone Project

An agentic RAG system built for the DataTalksClub LLM Zoomcamp 2026 cohort. Combines hybrid search (keyword + vector with Reciprocal Rank Fusion), iterative query reformulation, and a Streamlit chat interface with full agent transparency.

## Project Goal

Given a corpus of 3,200 Wikipedia passages and 918 Q&A pairs from the `rag-mini-wikipedia` dataset, build a question-answering system that:

1. Retrieves relevant documents using hybrid search
2. Uses an agentic loop to reformulate queries when results are insufficient
3. Generates faithful, grounded answers via a local LLM
4. Provides full transparency into the search and reasoning process
5. Tracks performance with OpenTelemetry-based monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                      │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Chat UI  │  │ Agent Visual │  │ Source Docs Display │    │
│  └──────────┘  └──────────────┘  └─────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Agent (agent.py)                     │
│                                                             │
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
│ HuggingFace ──► corpus.jsonl ──► chunker ──► documents.jsonl│
│  (rag-mini-        (3,200         (sentence     (chunked,   │
│   wikipedia)       passages)       splitting)    indexed)   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Monitoring (OpenTelemetry + SQLite)                 │
│                                                             │
│  TracerSetup ──► SQLiteSpanExporter ──► traces.db           │
│  (spans for         (custom schema)      (queries, tokens,  │
│   agent.run,                             feedback, latency) │
│   search, LLM)                                              │
│                                                             │
│  Dashboard: Streamlit (dashboard.py)                        │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### Quick Start (Docker)

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and OPENAI_API_BASE_URL to your LLM API endpoint (local hosted LLM or cloud OpenAI-compatible API)
docker-compose up --build
```

App available at `http://localhost:8501`.

### Local Development

```bash
uv sync
uv run python -m src.data.download_model   # fetches ONNX embedder (tokenizer.json + model.onnx)
cp .env.example .env
# Edit .env with your settings
uv run streamlit run src/interface/app.py
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

## Usage

```bash
# Ask a question via the Streamlit UI at http://localhost:8501

# Or run the agent from CLI:
set -a; source .env; set +a; uv run python -c "from src.rag.agent import RAGAgent; a = RAGAgent(); r = a.run('What is machine learning?'); print(r['answer'][:200])"

# Run evaluations:
uv run python -m src.evaluation.retrieval_eval
uv run python -m src.evaluation.llm_eval
uv run python -m src.evaluation.agent_eval

# Open monitoring dashboard:
uv run streamlit run src/monitoring/dashboard.py
```

See [docs/usage.md](docs/usage.md) for complete usage guide.

## Evaluation Results

### Retrieval (918 questions, top-5)

| Method    | Precision@5 | Recall@5 | MRR    | Time   |
|-----------|-------------|----------|--------|--------|
| Keyword   | 0.0004      | 0.0022   | 0.0014 | 1.1s   |
| Vector    | 0.0011      | 0.0054   | 0.0022 | 7.7s   |
| Hybrid    | 0.0009      | 0.0044   | 0.0021 | 7.5s   |

### LLM Answer Quality (10-question sample, 1-5 scale)

| Prompt       | Faithfulness | Relevance | Coherence | Time     |
|--------------|-------------|-----------|-----------|----------|
| Simple       | 4.9         | 5.0       | 5.0       | 173.6s   |
| Detailed     | 4.9         | 5.0       | 5.0       | 181.9s   |
| With Examples| 5.0         | 5.0       | 5.0       | 230.1s   |

### Agent vs Simple RAG

| Metric                  | Simple RAG | Agentic RAG |
|-------------------------|------------|-------------|
| Retrieval Hit Rate      | 0.44%      | 4.0%        |
| Avg Searches/Query      | 1.0        | 1.0         |
| Latency/Query           | 0.0s       | 0.01s       |
| Retrieval Improvement   | —          | +3.56%      |

See [docs/evaluation.md](docs/evaluation.md) for detailed evaluation methodology and results.

## Project Structure

```
project/
├── README.md
├── docker-compose.yml      # Docker orchestration (app + PostgreSQL)
├── Dockerfile              # App container (Python 3.13 + uv)
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
├── data/
│   ├── corpus.jsonl        # Raw Wikipedia passages (3,200)
│   ├── qa.jsonl            # Q&A pairs (918)
│   ├── chunks/
│   │   └── documents.jsonl # Chunked documents
│   └── traces.db           # Monitoring data (SQLite)
├── models/
│   └── Xenova/all-MiniLM-L6-v2/  # ONNX embedder (tokenizer.json + model.onnx)
├── docker/
│   └── entrypoint.sh       # Pipeline orchestration
├── results/
│   ├── retrieval_eval.json # Retrieval evaluation results
│   ├── llm_eval.json       # LLM evaluation results
│   ├── agent_eval.json     # Agent evaluation results
│   └── agent_eval_comparison.png
├── src/
│   ├── data/
│   │   ├── ingest.py       # HuggingFace dataset download
│   │   ├── chunker.py      # Document chunking pipeline
│   │   └── download_model.py  # ONNX embedding model download
│   ├── search/
│   │   ├── embedder.py     # ONNX embedder (onnxruntime, no torch)
│   │   └── hybrid.py       # Hybrid search (keyword + vector + RRF)
│   ├── rag/
│   │   ├── pipeline.py     # Base RAG pipeline
│   │   └── agent.py        # Agentic RAG with iterative reformulation
│   ├── interface/
│   │   └── app.py          # Streamlit chat UI
│   └── monitoring/
│       ├── tracer.py       # OpenTelemetry tracing + SQLite exporter
│       └── dashboard.py    # Monitoring dashboard
├── tests/                  # Test suite
└── notebooks/              # Exploration notebooks
```

## Key Dependencies

- **minsearch** — keyword search index (NOT vector search despite the name)
- **ONNX + tokenizers** — local embeddings (all-MiniLM-L6-v2 via onnxruntime, no torch)
- **OpenAI** — LLM calls via the Responses API to the configured endpoint (locally hosted LLM or cloud OpenAI-compatible API)
- **Streamlit** — chat interface and monitoring dashboard
- **OpenTelemetry** — distributed tracing with SQLite storage

## LLM Backend

LLM calls go through the OpenAI Responses API to the endpoint configured via `OPENAI_API_BASE_URL` in `.env` — either a locally hosted LLM (e.g. `http://localhost:9101/v1`) or a cloud OpenAI-compatible API (e.g. `https://api.openai.com/v1`). The endpoint must be reachable for any LLM call to work. Model: set `MODEL_ID` in `.env` (required, no default). In Docker, a `localhost` base URL is automatically rewritten to `host.docker.internal` so the container can reach a locally hosted LLM on the host; cloud URLs pass through unchanged.

All LLM calls use the OpenAI Responses API (`client.responses.create`), not Chat Completions.


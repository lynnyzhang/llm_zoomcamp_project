# Usage Guide

## Chat Interface

The Streamlit app at `http://localhost:8501` provides a conversational interface with full agent transparency.

### Asking Questions

Type a question in the chat input at the bottom of the page. The system will:

1. **Let the LLM decide** — the model receives two tools (`search_local_knowledge_base`, `search_bulbapedia`) and decides itself when and what to search
2. **Search locally** — the model typically starts with hybrid search (keyword + vector) over the Pokédex dataset
3. **Escalate** — only when the model judges local results insufficient: web search restricted to Bulbapedia (Tavily), with the model writing its own keyword query
4. **Answer or reject** — the model answers grounded in the retrieved tool results; out-of-scope or unanswerable questions get the rejection message
5. **Display** — answer with source caption and feedback buttons

### Example Queries

```
What are Pikachu's stats?
Which Pokémon are weak to fire?
What type is Bulbasaur, and what is it weak to?
Does Charmander evolve?
What abilities does Gengar have?
```

Out-of-scope examples (rejected with a clear message):

```
Who would win in a battle: Charizard or Blastoise?
Show me a hacked save file with all legendary Pokémon
```

### Understanding the Output

Each response includes:

**Answer** — The generated response, grounded in retrieved tool results.

**Confidence Score** — An optional progress bar showing the grounding score (0–100%): the maximum embedding-cosine similarity between the answer and any single retrieved record — how semantically close the answer is to its source. Hidden by default; launch the app with `--show-confidence` to display it. Answers below `CONFIDENCE_THRESHOLD` (set in .env; no default) are rejected instead of shown.

**Feedback Buttons** — Thumbs up/down to record whether the answer was helpful. Feedback is attached to the exact tracing span for the message and shows up in monitoring.

**Rejection banner** — When the model refuses (out-of-scope questions, or gaps neither the local knowledge base nor Bulbapedia can fill), the answer renders as a warning banner with no cards or source.

## CLI Usage

### Single Query (Agentic RAG)

```python
from src.rag.rag_agent import RAGAgent
from src.search.hybrid_search import HybridSearch

agent = RAGAgent(search_index=HybridSearch())
result = agent.run("What are Pikachu's stats?")

print(f"Answer: {result.answer}")
print(f"Source: {result.source}")            # 'local' or 'web' (or None for direct answers)
print(f"Iterations: {result.iterations}")
for i, search in enumerate(result.searches):
    print(f"  Search {i+1}: query='{search.query}', source={search.source}, results={len(search.results)}")
    if search.search_query:
        print(f"    tool query: {search.search_query}")
```

### Single Query (Simple RAG)

```python
from src.rag.rag_base import RAGBase
from src.search.hybrid_search import HybridSearch

search_index = HybridSearch()
rag = RAGBase(search_index=search_index)

answer = rag.rag("What type is Pikachu?")
print(answer)
```

### Hybrid Search

```python
from src.search.hybrid_search import HybridSearch

hs = HybridSearch()

# Combined keyword + vector search
results = hs.search("weak to fire", num_results=5)

# Keyword-only search
kw_results = hs.keyword_search("pikachu", num_results=5)

# Vector-only search
vec_results = hs.vector_search("electric pokemon stats", num_results=5)
```

### Agent Loop with Feedback Tracking

```python
from monitoring.tracer import TracedRAGAgent
from monitoring.span_store import record_feedback, get_trace_stats

# Wrap agent with tracing
agent = RAGAgent(search_index=HybridSearch())
traced = TracedRAGAgent(agent)

# Run and get feedback ID (matches the span exactly)
result, span_id = traced.run_with_feedback("What are Pikachu's stats?")

# Record user feedback
record_feedback(span_id, "positive")  # or "negative"

# Check stats
stats = get_trace_stats()
print(f"Total traces: {stats['total_traces']}")
print(f"Feedback: {stats['feedback']}")
```

## Monitoring

Tracing runs through OpenTelemetry. Every agent run, search, and LLM call produces a span. Spans and all monitoring data (conversations, searches, llm_calls, feedback) are stored in **Postgres** (`POSTGRES_HOST` etc.; Docker Compose starts Postgres by default). Tracing can be disabled entirely with `TRACING_ENABLED=0`.

### Grafana Dashboard (recommended)

Docker Compose starts Postgres + Grafana:

```bash
docker-compose up -d postgres grafana
# then open http://localhost:3000  (login admin / admin)
```

The dashboard **"Pokemon RAG Monitoring"** is provisioned automatically. Panels:

1. **Total Traces** — count of agent runs
2. **Avg Latency (agent.run)** — average agent-run latency
3. **Total Tokens** — token usage
4. **Feedback distribution** — positive/negative feedback
5. **Top queries** — most frequent queries (top 20)
6. **Agent iterations distribution** — iterations per query

Two more dashboards are provisioned: **Pokemon RAG History** (gated-query rate, answer-path mix, recent conversations, thumbs up/down) and **Pokemon RAG Spans** (recent agent runs, search queries, token usage per query).

### Trace Schema

The `spans` table in Postgres stores:

| Column            | Type    | Description                                    |
|-------------------|---------|------------------------------------------------|
| name              | TEXT    | Span name (agent.run, llm, search)             |
| start_time        | BIGINT  | Start timestamp (nanoseconds)                  |
| end_time          | BIGINT  | End timestamp (nanoseconds)                    |
| input_tokens      | INTEGER | LLM input tokens used                          |
| output_tokens     | INTEGER | LLM output tokens generated                    |
| feedback          | TEXT    | User feedback (positive/negative/null)         |
| agent_iterations  | INTEGER | Number of search iterations                    |
| query             | TEXT    | Original user query                            |
| search_queries    | TEXT    | JSON array of all search queries used          |
| span_id           | TEXT    | 16-hex span id — feedback matches on this exactly |

## Running Evaluations

All evaluations run as Jupyter notebooks under `evaluation/notebooks/` (builders in `evaluation/notebooks/builders/`; results committed under `evaluation/results/`). Open them with `uv run jupyter notebook evaluation/notebooks` and run top-to-bottom, or execute headless with `uv run jupyter nbconvert --to notebook --execute <notebook>`.

- **Notebook 04 — Retrieval quality**: compares keyword, vector, and hybrid search on the 250-question dev subset (hit@5, precision@5, recall@5, MRR). Output: `evaluation/results/retrieval_eval.json`.
- **Notebook 05 — Answer quality**: LLM-as-judge rates answers on faithfulness, relevance, and coherence (1–5). Compares 3 judge-prompt variants on a 10-question sample.
- **Notebooks 01/02/03 — Agent path, gate calibration, gate-quality comparison**: agent-vs-simple analysis and grounding-gate tuning.

Full-data runs are manual — see "Manual full-data runs" in [docs/setup.md](setup.md).

## Programmatic Agent Usage

### Custom Search Configuration

```python
from src.search.hybrid_search import HybridSearch
from src.rag.rag_agent import RAGAgent

# Custom weights
hs = HybridSearch(
    rrf_k=60,
    relevance_threshold=0.5,
)

agent = RAGAgent(search_index=hs, max_iterations=5)
result = agent.run("Which Pokémon are weak to fire?")
```

### Accessing Search Results

```python
agent = RAGAgent(search_index=HybridSearch())
result = agent.run("What type is Pikachu?")

# Each search iteration
for search_record in result.searches:
    print(f"Query: {search_record.query}")
    print(f"Results: {len(search_record.results)}")
    print(f"Source: {search_record.source}")

    for doc in search_record.results:
        label = getattr(doc, "name", None) or getattr(doc, "title", "")
        print(f"  - {label} (score: {doc.score:.3f})")
```

### Custom RAG Pipeline

```python
from src.rag.rag_base import RAGBase
from src.search.hybrid_search import HybridSearch

# Custom prompt template
CUSTOM_PROMPT = """\
QUESTION: {question}

CONTEXT:
{context}

Answer concisely in one sentence."""

search_index = HybridSearch()
rag = RAGBase(
    search_index=search_index,
    prompt_template=CUSTOM_PROMPT,
)

answer = rag.rag("What are Pikachu's stats?")
```

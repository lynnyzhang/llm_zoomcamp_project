# Usage Guide

## Chat Interface

The Streamlit app at `http://localhost:8501` provides a conversational interface with full agent transparency.

### Asking Questions

Type a question in the chat input at the bottom of the page. The system will:

1. **Let the LLM decide** — the model receives two tools (`search_local_knowledge_base`, `search_bulbapedia`) and decides itself when and what to search
2. **Search locally** — the model typically starts with hybrid search (keyword + vector) over the Pokédex corpus
3. **Escalate** — only when the model judges local results insufficient: web search restricted to Bulbapedia (Tavily), with the model writing its own keyword query
4. **Answer or reject** — the model answers grounded in the retrieved tool results; out-of-scope or unanswerable questions get the rejection message
5. **Display** — answer with source caption, Pokémon cards, and feedback buttons

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

**Confidence Score** — An optional progress bar showing the grounding score (0–100%): the maximum embedding-cosine similarity between the answer and any single retrieved record — how semantically close the answer is to its source. Hidden by default; launch the app with `--show-confidence` to display it. Answers below `CONFIDENCE_THRESHOLD` (default 0.65) are rejected instead of shown.

**Pokémon Cards** — Retrieved documents render as cards with official artwork (sprite URLs from the dataset, PokeAPI fallback), the Pokémon's name and types, and a stats excerpt. Only Pokémon named in the question are shown — questions that name no Pokémon get no cards. Artwork that fails to load degrades gracefully — the card still shows the title.

**Feedback Buttons** — Thumbs up/down to record whether the answer was helpful. Feedback is attached to the exact tracing span for the message and shows up in monitoring.

**Rejection banner** — When the model refuses (out-of-scope questions, or gaps neither the local knowledge base nor Bulbapedia can fill), the answer renders as a warning banner with no cards or source.

## CLI Usage

### Single Query (Agentic RAG)

```python
from src.rag.agent import RAGAgent

agent = RAGAgent()
result = agent.run("What are Pikachu's stats?")

print(f"Answer: {result['answer']}")
print(f"Source: {result['source']}")            # 'local' or 'web' (or None for direct answers)
print(f"Iterations: {result['iterations']}")
for i, search in enumerate(result['searches']):
    print(f"  Search {i+1}: query='{search.query}', source={search.source}, results={len(search.results)}")
    if search.reformulated_query:
        print(f"    tool query: {search.reformulated_query}")
```

### Single Query (Simple RAG)

```python
from src.rag.RAGBase import RAGBase
from src.search.hybrid import HybridSearch

search_index = HybridSearch()
rag = RAGBase(search_index=search_index)

answer = rag.rag("What type is Pikachu?")
print(answer)
```

### Hybrid Search

```python
from src.search.hybrid import HybridSearch

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
from monitoring.tracer import TracedRAGAgent, record_feedback, get_trace_stats

# Wrap agent with tracing
agent = RAGAgent()
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

Tracing runs through OpenTelemetry. Every agent run, search, and LLM call produces a span. Spans go to SQLite (`monitoring/traces.db`, always on) and optionally to Postgres when `POSTGRES_HOST` is set. Tracing can be disabled entirely with `TRACING_ENABLED=0`.

### Grafana Dashboard (recommended)

Docker Compose starts Postgres + Grafana:

```bash
docker-compose up -d postgres grafana
# then open http://localhost:3000  (login admin / admin)
```

The dashboard **"Pokemon RAG Monitoring"** is provisioned automatically. Panels:

1. **Stats row** — Total traces, total cost, average latency, total tokens
2. **Queries over time** — Line chart of agent-run volume
3. **Feedback distribution** — Bar chart of positive/negative feedback
4. **Latency over time** — Average agent-run latency
5. **Token usage over time** — Input vs output tokens
6. **Top queries** — Most frequent queries (top 20)
7. **Agent iterations distribution** — Iterations per query

### Streamlit Dashboard (SQLite)

For a lightweight dashboard over the local SQLite store:

```bash
uv run streamlit run monitoring/dashboard.py
```

Sections: summary metrics, queries over time, feedback distribution, latency distribution, token usage, popular topics, agent search patterns, raw trace data.

### Trace Schema

The `spans` table in `monitoring/traces.db` (mirrored in Postgres) stores:

| Column            | Type    | Description                                    |
|-------------------|---------|------------------------------------------------|
| name              | TEXT    | Span name (agent.run, llm, search)             |
| start_time        | INTEGER | Start timestamp (nanoseconds)                  |
| end_time          | INTEGER | End timestamp (nanoseconds)                    |
| input_tokens      | INTEGER | LLM input tokens used                          |
| output_tokens     | INTEGER | LLM output tokens generated                    |
| cost              | REAL    | Estimated cost                                 |
| feedback          | TEXT    | User feedback (positive/negative/null)         |
| agent_iterations  | INTEGER | Number of search iterations                    |
| query             | TEXT    | Original user query                            |
| search_queries    | TEXT    | JSON array of all search queries used          |
| span_id           | TEXT    | 16-hex span id — feedback matches on this exactly |

## Running Evaluations

### Retrieval Evaluation

Compares keyword, vector, and hybrid search on the dev-subset ground-truth questions (250).

```bash
uv run python -m evaluation.retrieval_eval
```

Output: `evaluation/results/retrieval_eval.json`

### LLM Answer Quality Evaluation

Uses LLM-as-judge to rate answers on faithfulness, relevance, and coherence (1-5 scale). Tests 3 judge prompt variants on a 10-question sample (the in-script `sample_size` constant).

```bash
uv run python -m evaluation.llm_eval
```

Output: `evaluation/results/llm_eval.json`

### Agent vs Simple RAG Evaluation

Compares agentic RAG (manual tool-use loop: the model decides when to search local / Bulbapedia web) against simple RAG (single search) on retrieval accuracy, answer quality, and latency.

```bash
uv run python -m evaluation.agent_eval
```

Output: `evaluation/results/agent_eval.json` and `evaluation/results/agent_eval_comparison.png`

Full-data runs are manual — see "Manual full-data runs" in [docs/setup.md](setup.md).

## Programmatic Agent Usage

### Custom Search Configuration

```python
from src.search.hybrid import HybridSearch
from src.rag.agent import RAGAgent

# Custom weights
hs = HybridSearch(
    rrf_k=60,
    rrf_k=60,
)

agent = RAGAgent(search_index=hs, max_iterations=5)
result = agent.run("Which Pokémon are weak to fire?")
```

### Accessing Search Results

```python
agent = RAGAgent()
result = agent.run("What type is Pikachu?")

# Each search iteration
for search_record in result['searches']:
    print(f"Query: {search_record.query}")
    print(f"Results: {len(search_record.results)}")
    print(f"Analysis: {search_record.analysis}")

    for doc in search_record.results:
        print(f"  - {doc['title']} (score: {doc.get('score', 'N/A')})")
```

### Custom RAG Pipeline

```python
from src.rag.RAGBase import RAGBase
from src.search.hybrid import HybridSearch

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

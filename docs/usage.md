# Usage Guide

## Chat Interface

The Streamlit app at `http://localhost:8501` provides a conversational interface with full agent transparency.

### Asking Questions

Type a question in the chat input at the bottom of the page. The system will:

1. **Search** the document corpus using hybrid search (keyword + vector)
2. **Analyze** whether results are sufficient to answer
3. **Reformulate** the query if results are insufficient (up to 3 iterations)
4. **Generate** a grounded answer from the retrieved context
5. **Display** the answer with confidence score and full process visualization

### Example Queries

```
What is machine learning?
How do I store vectors in PostgreSQL?
What is the difference between supervised and unsupervised learning?
Explain the transformer architecture
```

### Understanding the Output

Each response includes:

**Answer** — The generated response, grounded in retrieved documents.

**Confidence Score** — A progress bar showing how confident the system is. Calculated from:
- Proportion of search iterations that found sufficient results (70%)
- Efficiency bonus for finding results in fewer iterations (30%)

**Agent Process** — Expandable sections showing each search iteration:
- The query used (original or reformulated)
- Number of results found
- LLM analysis of result sufficiency
- Reformulation suggestion (if results were insufficient)
- Top search results with scores

**Source Documents** — Deduplicated list of all documents used across iterations, with:
- Document ID, title, section
- Relevance score
- Content preview

**Feedback Buttons** — Thumbs up/down to record whether the answer was helpful.

### Sidebar Settings

| Setting              | Default | Range  | Description                                      |
|---------------------|---------|--------|--------------------------------------------------|
| Search results       | 5       | 1-10   | Number of documents retrieved per search iteration|
| Search type          | hybrid  | —      | hybrid / keyword / vector                         |
| Max agent iterations | 3       | 1-5    | Maximum search-reformulate cycles                 |

Changing settings takes effect on the next query.

## CLI Usage

### Single Query (Agentic RAG)

```python
from src.rag.agent import RAGAgent

agent = RAGAgent()
result = agent.run("What is machine learning?")

print(f"Answer: {result['answer']}")
print(f"Iterations: {result['iterations']}")
for i, search in enumerate(result['searches']):
    print(f"  Search {i+1}: query='{search.query}', results={len(search.results)}")
    print(f"    Sufficient: {search.analysis.get('sufficient')}")
```

### Single Query (Simple RAG)

```python
from src.rag.pipeline import RAGBase
from src.search.hybrid import HybridSearch

search_index = HybridSearch()
rag = RAGBase(search_index=search_index)

answer = rag.rag("What is machine learning?")
print(answer)
```

### Hybrid Search

```python
from src.search.hybrid import HybridSearch

hs = HybridSearch()

# Combined keyword + vector search
results = hs.search("machine learning", num_results=5)

# Keyword-only search
kw_results = hs.keyword_search("machine learning", num_results=5)

# Vector-only search
vec_results = hs.vector_search("machine learning", num_results=5)
```

### Agent Loop with Feedback Tracking

```python
from src.monitoring.tracer import TracedRAGAgent, record_feedback, get_trace_stats

# Wrap agent with tracing
agent = RAGAgent()
traced = TracedRAGAgent(agent)

# Run and get feedback ID
result, span_id = traced.run_with_feedback("What is deep learning?")

# Record user feedback
record_feedback(span_id, "positive")  # or "negative"

# Check stats
stats = get_trace_stats()
print(f"Total traces: {stats['total_traces']}")
print(f"Feedback: {stats['feedback']}")
```

## Monitoring Dashboard

Start the monitoring dashboard:

```bash
# Docker
docker-compose up app  # Access at http://localhost:8501

# Local
uv run streamlit run src/monitoring/dashboard.py
```

### Dashboard Sections

1. **Summary Metrics** — Total traces, LLM calls, cost, average latency
2. **Queries Over Time** — Line chart of query volume (auto-refreshes every 30s)
3. **Feedback Distribution** — Bar chart of positive/negative feedback
4. **Latency Distribution** — Duration statistics by span type + histogram
5. **Token Usage** — Input vs output token counts per span
6. **Popular Topics** — Top 20 most frequent queries
7. **Agent Search Patterns** — Iterations per query, iteration distribution, recent search patterns
8. **Raw Trace Data** — Full trace table (expandable)

### Trace Schema

The `spans` table in `data/traces.db` stores:

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

## Running Evaluations

### Retrieval Evaluation

Compares keyword, vector, and hybrid search on 918 Q&A pairs.

```bash
uv run python -m src.evaluation.retrieval_eval
```

Output: `results/retrieval_eval.json`

### LLM Answer Quality Evaluation

Uses LLM-as-judge to rate answers on faithfulness, relevance, and coherence (1-5 scale). Tests 3 judge prompt variants.

```bash
uv run python -m src.evaluation.llm_eval
```

Output: `results/llm_eval.json`

### Agent vs Simple RAG Evaluation

Compares agentic RAG (iterative search) against simple RAG (single search) on retrieval accuracy, answer quality, and latency.

```bash
uv run python -m src.evaluation.agent_eval
```

Output: `results/agent_eval.json` and `results/agent_eval_comparison.png`

## Programmatic Agent Usage

### Custom Search Configuration

```python
from src.search.hybrid import HybridSearch
from src.rag.agent import RAGAgent

# Custom weights
hs = HybridSearch(
    keyword_weight=1.0,
    vector_weight=0.7,
    rrf_k=60,
)

agent = RAGAgent(search_index=hs, max_iterations=5)
result = agent.run("Your question here")
```

### Accessing Search Results

```python
agent = RAGAgent()
result = agent.run("What is transformer?")

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
from src.rag.pipeline import RAGBase, PROMPT_TEMPLATE
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
    model="qwen/qwen3.5-9b",
)

answer = rag.rag("What is machine learning?")
```

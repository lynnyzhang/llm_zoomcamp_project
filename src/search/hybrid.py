"""
Hybrid search combining keyword (minsearch) and vector (ONNX embedder)
indices with Reciprocal Rank Fusion.
"""

import json
from pathlib import Path

from minsearch import Index, VectorSearch

from src.search.embedder import Embedder


def _load_documents(path: str | Path) -> list[dict]:
    """Load documents from a JSONL file."""
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
    num_results: int = 5,
) -> list[dict]:
    """Combine multiple ranked result lists using weighted Reciprocal Rank Fusion.

    For each document appearing in any result list, its score is:
        sum(weight_i * 1 / (k + rank_i))

    where rank_i is 0-based position in list i, and weight_i is the
    weight for that list.

    Args:
        result_lists: List of ranked result lists (each a list of dicts).
        weights: Per-list weights. Defaults to equal weights (1.0 each).
        k: RRF constant (higher = less weight to top ranks). Default 60.
        num_results: How many results to return.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict[tuple, float] = {}
    docs: dict[tuple, dict] = {}

    for weight, results in zip(weights, result_lists):
        for rank, doc in enumerate(results):
            key = (doc["id"],)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    results = []
    for key in ranked[:num_results]:
        doc = dict(docs[key])  # shallow copy to avoid mutating originals
        doc["score"] = scores[key]
        results.append(doc)
    return results


class HybridSearch:
    """Hybrid search index combining keyword search and vector search.

    Keyword search uses minsearch.Index on content/title fields.
    Vector search uses the ONNX all-MiniLM-L6-v2 embedder (onnxruntime,
    no torch) with minsearch.VectorSearch.

    Results are fused using weighted Reciprocal Rank Fusion (RRF).
    """

    # Default data path relative to this file
    _DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "chunks" / "documents.jsonl"

    def __init__(
        self,
        documents_path: str | Path | None = None,
        documents: list[dict] | None = None,
        model_path: str | Path | None = None,
        keyword_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        """
        Args:
            documents_path: Path to JSONL file with documents.
                Defaults to project/data/chunks/documents.jsonl.
            documents: Pre-loaded documents (overrides documents_path).
            model_path: Directory with the ONNX model (tokenizer.json +
                model.onnx). Defaults to the standard model location.
            keyword_weight: Weight for keyword search in RRF fusion.
            vector_weight: Weight for vector search in RRF fusion.
            rrf_k: RRF constant (higher = less rank sensitivity).
        """
        if documents is not None:
            self.documents = documents
        elif documents_path is not None:
            self.documents = _load_documents(documents_path)
        else:
            self.documents = _load_documents(self._DEFAULT_DATA)

        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

        # --- keyword index ---
        self.keyword_index = Index(
            text_fields=["content", "title"],
            keyword_fields=["section", "id"],
        )
        self.keyword_index.fit(self.documents)

        # --- vector index ---
        self.embedder = Embedder(model_path)
        texts = [doc["content"] for doc in self.documents]
        self.embeddings = self.embedder.encode_batch(texts, normalize=True)
        self.vector_index = VectorSearch()
        self.vector_index.fit(self.embeddings, self.documents)

    def search(
        self,
        query: str,
        num_results: int = 5,
        keyword_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> list[dict]:
        """Search combining keyword and vector results via RRF.

        Args:
            query: Search query string.
            num_results: Number of results to return.
            keyword_weight: Override default keyword weight.
            vector_weight: Override default vector weight.

        Returns:
            List of document dicts sorted by fused RRF score, each with
            an added 'score' field.
        """
        kw = keyword_weight if keyword_weight is not None else self.keyword_weight
        vw = vector_weight if vector_weight is not None else self.vector_weight

        # keyword search
        keyword_results = self.keyword_index.search(
            query, num_results=num_results * 2
        )

        # vector search
        query_vector = self.embedder.encode(query, normalize=True)
        vector_results = self.vector_index.search(
            query_vector, num_results=num_results * 2
        )

        # fuse with RRF
        fused = reciprocal_rank_fusion(
            [keyword_results, vector_results],
            weights=[kw, vw],
            k=self.rrf_k,
            num_results=num_results,
        )

        return fused

    def keyword_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Keyword-only search via minsearch."""
        return self.keyword_index.search(query, num_results=num_results)

    def vector_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Vector-only search via minsearch VectorSearch."""
        query_vector = self.embedder.encode(query, normalize=True)
        return self.vector_index.search(query_vector, num_results=num_results)


# ---------------------------------------------------------------------------
# CLI: build indices and test search
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "chunks" / "documents.jsonl"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading documents from {data_path}...")
    hybrid = HybridSearch(documents_path=data_path, vector_weight=0.7, keyword_weight=1.0)

    test_queries = [
        "pikachu",
        "Which Pokémon are weak to fire?",
        "electric pokemon stats",
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        results = hybrid.search(q, num_results=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['id']}] {r['title'][:80]}  (score: {r.get('score', 'N/A'):.6f})")

    print("\nHybrid search index built and tested successfully.")

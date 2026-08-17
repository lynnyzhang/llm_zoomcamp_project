import json
import os
from pathlib import Path

from minsearch import Index, VectorSearch

from src.search.embedder import Embedder
from src.search.reranker import Reranker
from src.search.search_records import PokemonDoc, parse_doc


def load_documents(path: Path | str) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def get_relevance_threshold():
    return float(os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.3"))


# Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank_i)
def rrf(
    result_lists: list[list[dict]], k: int = 60, num_results: int = 5
) -> list[dict]:
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc.get("chunk_id") or doc["id"],)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    results = []
    for key in ranked[:num_results]:
        doc = dict(docs[key])  # shallow copy to avoid mutating originals
        doc["score"] = scores[key]
        results.append(doc)
    return results


class HybridSearch:
    DEFAULT_DATA = (
        Path(__file__).resolve().parents[2] / "data" / "chunks" / "documents.jsonl"
    )

    def __init__(
        self,
        documents_path: Path | str | None = None,
        rrf_k: int = 60,
        relevance_threshold: float | None = None,
    ):
        if documents_path is not None:
            self.documents = load_documents(documents_path)
        else:
            self.documents = load_documents(self.DEFAULT_DATA)

        self.rrf_k = rrf_k
        self.relevance_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else get_relevance_threshold()
        )

        self.keyword_index = Index(
            text_fields=["search_text", "name"],
            keyword_fields=["id", "types", "kind"],
        )
        self.keyword_index.fit(self.documents)

        self.embedder = Embedder()
        texts = [doc["search_text"] for doc in self.documents]
        self.embeddings = self.embedder.encode_batch(texts, normalize=True)
        self.vector_index = VectorSearch()
        self.vector_index.fit(self.embeddings, self.documents)

        self.reranker = Reranker()

    def _filter_relevance(self, docs, query_vector):
        # The RRF score is rank-based, not a cosine, so relevance is the
        # query↔chunk cosine (the vector-search score). Compute it for every
        # chunk and drop those below the threshold; a chunk with no cosine
        # (keyword-only) is kept conservatively.
        cosines = self.embeddings @ query_vector
        chunk_cosine = {
            doc.get("chunk_id") or doc["id"]: float(c)
            for doc, c in zip(self.documents, cosines)
        }
        kept = []
        for doc in docs:
            cosine = chunk_cosine.get(doc.get("chunk_id") or doc["id"])
            if cosine is None or cosine >= self.relevance_threshold:
                kept.append(doc)
        return kept

    def search(self, query: str, num_results: int = 5) -> list[PokemonDoc]:
        keyword_results = self.keyword_index.search(query, num_results=num_results * 2)

        query_vector = self.embedder.encode(query, normalize=True)
        vector_results = self.vector_index.search(
            query_vector, num_results=num_results * 2
        )

        fused = rrf(
            [keyword_results, vector_results],
            k=self.rrf_k,
            num_results=num_results,
        )

        fused = self.reranker.rerank(query, fused)

        return [parse_doc(d) for d in self._filter_relevance(fused, query_vector)]

    def keyword_search(self, query: str, num_results: int = 5) -> list[PokemonDoc]:
        # No cosine is available for keyword-only results, so no relevance
        # filter applies here.
        results = self.keyword_index.search(query, num_results=num_results)
        return [parse_doc(d) for d in results]

    def vector_search(self, query: str, num_results: int = 5) -> list[PokemonDoc]:
        query_vector = self.embedder.encode(query, normalize=True)
        results = self.vector_index.search(query_vector, num_results=num_results)
        return [parse_doc(d) for d in self._filter_relevance(results, query_vector)]


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
    hybrid = HybridSearch(documents_path=data_path)

    test_queries = [
        "pikachu",
        "Which Pokémon are weak to fire?",
        "electric pokemon stats",
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        results = hybrid.search(q, num_results=3)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r.id}] {(r.name or r.search_text)[:80]}  (score: {(r.score or 'N/A'):.6f})"
            )

    print("\nHybrid search index built and tested successfully.")

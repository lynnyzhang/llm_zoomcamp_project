import os
from dataclasses import dataclass, field

from src.rag.llm_call_record import LLMCallSummary, Usage
from src.rag.prompts import REJECTION_MESSAGE
from src.search.embedder import Embedder
from src.rag.tools import SearchRecord


@dataclass
class AgentResult:
    answer: str
    searches: list[SearchRecord]
    iterations: int
    rejected: bool
    source: str | None
    confidence: float | None
    relevance: float | None
    usage: Usage = field(default_factory=Usage)
    llm_calls: list[LLMCallSummary] = field(default_factory=list)

    @classmethod
    def rejected_result(cls, searches):
        return cls(
            answer=REJECTION_MESSAGE,
            searches=searches,
            iterations=len(searches),
            rejected=True,
            source=None,
            confidence=None,
            relevance=None,
        )


def get_confidence_threshold():
    return float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))


def cosine_similarity(embedder: Embedder, text_a: str, text_b: str) -> float:
    """Cosine similarity of two texts' embedding vectors (0..1).

    Used for the two RAG quality scores: grounding (answer vs retrieved
    context — faithfulness) and relevance (question vs answer)."""
    if not text_a or not text_b:
        return 0.0
    va = embedder.encode(text_a, normalize=False)
    vb = embedder.encode(text_b, normalize=False)
    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = sum(x * x for x in va) ** 0.5
    norm_b = sum(y * y for y in vb) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return float(dot / (norm_a * norm_b))


def grounding_scores(
    embedder: Embedder, answer: str, searches: list[SearchRecord]
) -> list[float]:
    scores = []
    for record in searches:
        for item in record.results:
            text = getattr(item, "search_text", "") or getattr(item, "snippet", "")
            if text:
                scores.append(cosine_similarity(embedder, answer, text))
    return scores


def relevance_score(embedder: Embedder, query: str, answer: str) -> float:
    return cosine_similarity(embedder, query, answer)


def finalize_result(
    answer_text: str,
    query: str,
    searches: list[SearchRecord],
    sources: set[str],
    embedder: Embedder | None,
    confidence_threshold: float,
) -> AgentResult:
    answer = (answer_text or "").strip()
    result = AgentResult.rejected_result(searches)
    if not answer or answer == REJECTION_MESSAGE:
        return result
    confidence = None
    relevance = None
    if embedder is not None:
        # Grounding = the best match against any single retrieved
        # record — concatenating all records dilutes the answer's
        # actual source. Relevance = the question vs the answer.
        scores = grounding_scores(embedder, answer, searches)
        confidence = max(scores) if scores else 0.0
        relevance = relevance_score(embedder, query, answer)
    if confidence is not None and confidence < confidence_threshold:
        # Ungrounded answers (memory, invented facts, tool-less replies)
        # fail the gate — never surface them.
        return result
    result.answer = answer
    result.rejected = False
    result.source = (
        "web" if "web" in sources else ("local" if "local" in sources else None)
    )
    result.confidence = confidence
    result.relevance = relevance
    return result

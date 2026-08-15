import os

from src.rag.prompts import REJECTION_MESSAGE


def get_confidence_threshold():
    return float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))


def cosine_similarity(embedder, text_a, text_b):
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


def grounding_scores(embedder, answer, searches):
    scores = []
    for record in searches:
        for item in record.results:
            text = item.get("search_text") or item.get("snippet") or ""
            if text:
                scores.append(cosine_similarity(embedder, answer, text))
    return scores


def relevance_score(embedder, query, answer):
    return cosine_similarity(embedder, query, answer)


def finalize_result(answer_text, query, searches, sources,
                    embedder, confidence_threshold):
    answer = (answer_text or "").strip()
    if not answer or answer == REJECTION_MESSAGE:
        return {
            "answer": REJECTION_MESSAGE,
            "searches": searches,
            "iterations": len(searches),
            "rejected": True,
            "source": None,
            "confidence": None,
            "relevance": None,
        }
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
        return {
            "answer": REJECTION_MESSAGE,
            "searches": searches,
            "iterations": len(searches),
            "rejected": True,
            "source": None,
            "confidence": None,
            "relevance": None,
        }
    return {
        "answer": answer,
        "searches": searches,
        "iterations": len(searches),
        "rejected": False,
        "source": "web" if "web" in sources else ("local" if "local" in sources else None),
        "confidence": confidence,
        "relevance": relevance,
    }

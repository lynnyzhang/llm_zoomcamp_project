import json
from pathlib import Path

DEFAULT_DOCUMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "chunks" / "documents.jsonl"
)


def load_document_index(documents_path=None):
    """data/chunks/documents.jsonl -> {str(doc id): doc}.

    The ground-truth answer for a QA row is the linked document's content —
    the analog of the FAQ 'answer' lookup in the course's RAG eval
    (answer_orig = doc_idx[doc_id]["answer"]).
    """
    path = Path(documents_path) if documents_path else DEFAULT_DOCUMENTS_PATH
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                docs[str(doc["id"])] = doc
    return docs


def ground_truth_answer(doc_idx, document_id):
    """The linked document's content — the ground-truth answer for a QA row.

    Pokémon documents carry the answer inside search_text (the labeled record
    rendering), the analog of the FAQ 'answer' field in the course's evals.
    """
    doc = doc_idx.get(str(document_id))
    return doc.get("search_text") if doc else None


def load_pokemon_documents():
    """data/chunks/documents.jsonl → {int id: Pokémon document}.

    The indexed documents are the ground truth for question generation (the
    course's pattern generates questions from the indexed dataset, not from raw
    data). Type-chart docs (kind == "type_chart", string ids) are excluded —
    QA is per Pokémon.
    """
    if not DEFAULT_DOCUMENTS_PATH.exists():
        raise SystemExit(
            f"FATAL: indexed documents missing at {DEFAULT_DOCUMENTS_PATH}. "
            "Run `uv run python -m src.data.build_documents` first."
        )
    docs = {}
    with open(DEFAULT_DOCUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("kind") == "type_chart":
                continue
            docs[doc["id"]] = doc
    return docs

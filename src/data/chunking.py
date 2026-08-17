"""Token-aware sliding-window chunking of the search corpus.

The embedding model truncates at 128 tokens, so each Pokémon's full search_text
(254-265 tokens) loses its tail (evolution, type effectiveness, flavor text) to
retrieval and grounding. This mirrors the course's char-based chunk_documents
scheme but chunks at token boundaries so every chunk fits the embedding window.
"""

# Only the fields RAG actually consumes are kept on each chunk; `id` is the
# reference back to the original Pokémon doc (the course's filename analog).
CHUNK_FIELDS = ("id", "name", "types", "kind", "chunk_id", "start", "search_text")


def sliding_window_tokens(text, tokenizer, size=100, step=50):
    """Split text into overlapping token windows; returns {"start", "content"}.

    `start` is the token offset of the window's first token; `content` is the
    actual text slice (readable by the LLM), sliced via the encoding's per-token
    char offsets. Truncation/padding are disabled so the full token stream is
    available for offset slicing.
    """
    tokenizer.no_truncation()
    tokenizer.no_padding()
    # Special tokens ([CLS]/[SEP]) carry a zero-span (0,0) offset; drop them so
    # the last window slices real content instead of an empty span.
    offsets = [o for o in tokenizer.encode(text).offsets if o[0] != o[1]]
    n = len(offsets)
    chunks = []
    for start in range(0, n, step):
        end = min(start + size, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append({"start": start, "content": text[char_start:char_end]})
        if end == n:
            break
    return chunks


def chunk_documents(
    documents, tokenizer, size=100, step=50, content_field_name="search_text"
):
    """Chunk each document's content field, keeping only the search-relevant keys.

    Each chunk carries the parent's `id` (the reference to the original doc),
    `name`, `types`, plus {"chunk_id", "start", content_field}. All other parent
    metadata is dropped. `chunk_id` = f"{doc_id}:{start}" (token offset) for RRF
    dedup.
    """
    results = []
    for doc in documents:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop(content_field_name)
        chunks = sliding_window_tokens(doc_content, tokenizer, size=size, step=step)
        for chunk in chunks:
            chunk.update(doc_copy)
            chunk["chunk_id"] = f"{doc['id']}:{chunk['start']}"
            chunk[content_field_name] = chunk.pop("content")
            results.append({k: chunk[k] for k in CHUNK_FIELDS if k in chunk})
    return results

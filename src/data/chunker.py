"""Chunking and preprocessing pipeline for LLM Zoomcamp capstone.

Reads corpus.jsonl, applies optional re-chunking if passages exceed token limits,
adds metadata, and saves chunked documents to data/chunks/documents.jsonl.
"""

import json
import os
from pathlib import Path
from typing import Generator

# Token estimation: ~4 chars per token (rough approximation for English text)
CHARS_PER_TOKEN = 4
MIN_TOKENS = 500
MAX_TOKENS = 1000


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count (rough approximation)."""
    return len(text) // CHARS_PER_TOKEN


def re_chunk_passage(passage: str, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Re-chunk a passage if it exceeds max_tokens.
    
    Splits on sentence boundaries. Merges small sentences until they form
    chunks of at least min_tokens, but never exceeds max_tokens.
    
    Returns list of chunks. If passage is already within limits, returns [passage].
    """
    tokens = estimate_tokens(passage)
    
    # If within limits, return as-is
    if tokens <= max_tokens:
        return [passage]
    
    # Split on sentence boundaries
    sentences = []
    current = []
    for char in passage:
        current.append(char)
        if char in '.!?' and len(current) > 10:
            sentences.append(''.join(current).strip())
            current = []
    if current:
        sentences.append(''.join(current).strip())
    
    # Merge sentences into chunks
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If adding this sentence would exceed max, finalize current chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Merge very small final chunks
    if len(chunks) > 1 and estimate_tokens(chunks[-1]) < min_tokens:
        chunks[-2] = chunks[-2] + ' ' + chunks[-1]
        chunks.pop()
    
    return chunks if chunks else [passage]


def generate_metadata(passage_id: int, chunk_index: int, total_chunks: int) -> dict:
    """Generate metadata for a chunk.
    
    Since corpus.jsonl only has passage and id, we derive minimal metadata.
    The title/section/url can be enriched later if source data becomes available.
    """
    return {
        'title': f'Passage {passage_id}',
        'section': 'llm-zoomcamp',
        'url': f'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/passage/{passage_id}'
    }


def process_corpus(input_path: Path, output_path: Path) -> Generator[dict, None, None]:
    """Process corpus.jsonl and yield chunked documents."""
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            passage = record['passage']
            passage_id = record['id']
            
            # Apply re-chunking if needed
            chunks = re_chunk_passage(passage)
            
            # Yield each chunk as a document
            for chunk_index, chunk in enumerate(chunks):
                metadata = generate_metadata(passage_id, chunk_index, len(chunks))
                doc_id = f'{passage_id}_{chunk_index}' if len(chunks) > 1 else str(passage_id)
                
                yield {
                    'id': doc_id,
                    'title': metadata['title'],
                    'content': chunk,
                    'section': metadata['section'],
                    'url': metadata['url']
                }


def main():
    """Main entry point for the chunker pipeline."""
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / 'data' / 'corpus.jsonl'
    output_path = project_root / 'data' / 'chunks' / 'documents.jsonl'
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process and save
    chunk_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in process_corpus(input_path, output_path):
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            chunk_count += 1
    
    print(f'Processed {input_path}')
    print(f'Saved {chunk_count} chunks to {output_path}')
    
    # Quick validation
    with open(output_path, 'r', encoding='utf-8') as f:
        first_doc = json.loads(f.readline())
        print(f'Sample document keys: {list(first_doc.keys())}')


if __name__ == '__main__':
    main()

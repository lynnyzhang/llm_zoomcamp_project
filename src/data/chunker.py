import json
import threading
from pathlib import Path

# Token estimation: ~4 chars per token (rough approximation for English text)
CHARS_PER_TOKEN = 4
MIN_TOKENS = 500
MAX_TOKENS = 1000

# Official artwork URL pattern (PokeAPI sprites; verified 200 for ids 1..1025).
ARTWORK_URL = (
    'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/'
    'pokemon/other/official-artwork/{pokemon_id}.png'
)

_POKEDEX_PATH = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'complete_pokedex.json'
_POKEDEX = None
_POKEDEX_LOCK = threading.Lock()


def _get_pokedex():
    global _POKEDEX
    if _POKEDEX is None:
        with _POKEDEX_LOCK:
            if _POKEDEX is None:
                try:
                    with open(_POKEDEX_PATH, 'r', encoding='utf-8') as f:
                        _POKEDEX = {record['id']: record for record in json.load(f)}
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f'Missing raw pokedex cache at {_POKEDEX_PATH}. '
                        'Run `uv run python -m src.data.ingest` first.'
                    ) from exc
    return _POKEDEX


def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN


def re_chunk_passage(passage, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS):
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


def generate_metadata(passage_id, chunk_index, total_chunks):
    record = _get_pokedex().get(passage_id)
    if record is None:
        return {
            'title': f'Pokémon #{passage_id}',
            'section': 'unknown',
            'url': ARTWORK_URL.format(pokemon_id=passage_id),
        }
    types = '+'.join(record.get('types', []))
    section = types if types else record.get('generation', 'unknown')
    return {
        'title': f"{record['name'].capitalize()} (#{passage_id})",
        'section': section,
        'url': ARTWORK_URL.format(pokemon_id=passage_id),
    }


def process_corpus(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            passage = record['passage']
            passage_id = record['id']

            # A re-chunk split would yield suffixed ids "{id}_{n}" and break
            # eval exact-id linkage (review M4) - never split: keep ONE doc
            # per Pokémon (truncated to fit) so every id is a pure integer.
            chunks = re_chunk_passage(passage)
            if len(chunks) > 1:
                chunks = [chunks[0]]

            for chunk_index, chunk in enumerate(chunks):
                metadata = generate_metadata(passage_id, chunk_index, len(chunks))
                yield {
                    'id': str(passage_id),
                    'title': metadata['title'],
                    'content': chunk,
                    'section': metadata['section'],
                    'url': metadata['url'],
                }


def main():
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / 'data' / 'corpus.jsonl'
    output_path = project_root / 'data' / 'chunks' / 'documents.jsonl'

    if not input_path.exists():
        raise FileNotFoundError(
            f'Missing corpus at {input_path}. Run `uv run python -m src.data.ingest` first.'
        )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process and save
    docs = list(process_corpus(input_path, output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(doc, ensure_ascii=False) + '\n' for doc in docs)

    print(f'Processed {input_path}')
    print(f'Saved {len(docs)} chunks to {output_path}')

    # Quick validation
    with open(output_path, 'r', encoding='utf-8') as f:
        first_doc = json.loads(f.readline())
        print(f'Sample document keys: {list(first_doc.keys())}')


if __name__ == '__main__':
    main()

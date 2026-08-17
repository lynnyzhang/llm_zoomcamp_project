"""Build data/chunks/documents.jsonl, the hybrid-search corpus, from the two
bundled raw CSVs (pokemon_complete.csv + pokemon_types.csv)."""

import argparse
import json
import sys

from tokenizers import Tokenizer

from src.data.chunking import chunk_documents
from src.data.csv_parsers import load_raw_rows, parse_row
from src.data.download import DATA_DIR, TYPES_CSV
from src.data.evolution import EvolutionChain
from src.data.pokemon_doc_builder import PokemonDocBuilder
from src.data.type_chart import TypeChart
from src.search.embedder import default_model_dir

DOCUMENTS_PATH = DATA_DIR / "chunks" / "documents.jsonl"


def build_records(rows, limit):
    records = sorted((parse_row(r) for r in rows), key=lambda r: r["id"])
    return records if limit is None else records[:limit]


def load_tokenizer():
    tokenizer_path = default_model_dir() / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Embedding tokenizer not found at {tokenizer_path}. "
            "Run `uv run python -m src.data.download_model` to fetch it."
        )
    return Tokenizer.from_file(str(tokenizer_path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build the hybrid-search corpus documents.jsonl from the bundled "
            "CSVs (full 1,350 Pokémon by default; --limit for a subset)."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N Pokémon records by id (default: all 1,350)",
    )
    args = parser.parse_args(argv)

    records = build_records(load_raw_rows(), args.limit)
    chart = TypeChart.load(TYPES_CSV)
    chains = EvolutionChain.build_map(records)
    builder = PokemonDocBuilder()

    pokemon_docs = [
        builder.build(record, chart, chains.get(record["evolution_chain_id"]))
        for record in records
    ]
    chunks = chunk_documents(pokemon_docs, load_tokenizer(), size=100, step=50)
    charts = [chart.doc(t) for t in chart.chart]
    docs = chunks + charts

    DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs)

    print(
        f"Wrote {len(docs)} docs ({len(pokemon_docs)} Pokémon → {len(chunks)} chunks "
        f"+ {len(charts)} type charts) to {DOCUMENTS_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

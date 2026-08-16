import json
from pathlib import Path

from src.data.pokemon_doc_builder import PokemonDocBuilder
from src.data.evolution import EvolutionChain
from src.data.type_chart import TypeChart

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POKEMON_PATH = PROJECT_ROOT / "data" / "pokemon.jsonl"
TYPES_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "pokemon_types.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "chunks" / "documents.jsonl"


def main():
    if not POKEMON_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {POKEMON_PATH}. "
            "Run `uv run python -m src.data.ingest` first."
        )
    if not TYPES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing type chart at {TYPES_CSV_PATH}. "
            "Run `uv run python -m src.data.ingest` first."
        )

    with open(POKEMON_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    chart = TypeChart.load(TYPES_CSV_PATH)
    chains = EvolutionChain.build_map(records)
    builder = PokemonDocBuilder()

    docs = [
        builder.build(record, chart, chains.get(record["evolution_chain_id"]))
        for record in records
    ]
    # Part B: 18 type-chart docs appended after the 1,350 Pokémon docs,
    # in the order the rows appear in pokemon_types.csv (dict preserves order).
    for t in chart.chart:
        docs.append(chart.doc(t))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs)

    print(f"Processed {POKEMON_PATH}")
    print(f"Saved {len(docs)} docs to {OUTPUT_PATH}")

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"Sample document keys: {list(first.keys())}")


if __name__ == "__main__":
    main()

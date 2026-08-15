import argparse
import json
import sys

from src.data.csv_parsers import load_raw_rows, parse_row
from src.data.download import DATA_DIR

POKEMON_FILE = DATA_DIR / "pokemon.jsonl"


def build_records(rows, limit):
    parsed = [parse_row(r) for r in rows]
    parsed.sort(key=lambda r: r["id"])
    if limit is not None:
        parsed = parsed[:limit]
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build data/pokemon.jsonl from the Pokémon CSV dataset "
            "(patelris/pokemon-dataset-with-stats-and-types). "
            "Defaults to the FULL 1,350-record dataset."
        )
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Write only the first N records by id (default: all 1,350; "
             "dev runs typically use --limit 50)",
    )
    args = parser.parse_args(argv)

    rows = load_raw_rows()
    records = build_records(rows, args.limit)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(POKEMON_FILE, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
    print(f"Wrote {len(records)} records to {POKEMON_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

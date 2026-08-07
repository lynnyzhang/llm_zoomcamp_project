"""
Data ingestion script for the Complete Pokémon Dataset (Kaggle: elroytan/pokemondata).

Downloads the dataset archive anonymously via the Kaggle dataset API endpoint
(GET-only — HEAD returns 404 on this endpoint), inspects the archive before
parsing (schema_inspection.json), persists the FULL raw dataset to
data/raw/complete_pokedex.json, and builds data/corpus.jsonl with the exact
contract {"id": int, "passage": str} — one structured passage per Pokémon.

Default (dev subset): --limit 50 → the first 50 records by id.
--full: all 1,025 records. --limit N: any N.

The raw cache is the deterministic source for downstream todos (chunker
metadata, QA generation): the corpus is a derived slice of it and is never a
re-download.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

KAGGLE_DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/elroytan/pokemondata"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_POKEDEX = RAW_DIR / "complete_pokedex.json"
SCHEMA_INSPECTION = RAW_DIR / "schema_inspection.json"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"

# Required minimum field set — anything else must be mapped via .get() fallbacks.
REQUIRED_FIELDS = ("id", "name", "types", "stats", "damage_taken")

STAT_KEYS = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
DAMAGE_TAKEN_KEYS = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "steel",
    "dark", "fairy",
]


def download_archive(url: str = KAGGLE_DATASET_URL) -> zipfile.ZipFile:
    """GET the Kaggle dataset archive, following the 302 → signed GCS redirect.

    Must always use GET: this endpoint returns 404 for HEAD requests. The
    signed GCS URL is never hardcoded — we always go through the API endpoint.
    """
    print(f"Downloading archive from {url} ...")
    try:
        resp = requests.get(url, allow_redirects=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"FATAL: archive download failed: {exc}") from exc
    print(f"  HTTP {resp.status_code}, {len(resp.content)} bytes")
    return zipfile.ZipFile(io.BytesIO(resp.content))


def inspect_archive(zf: zipfile.ZipFile) -> str:
    """List the archive contents (inspect-first) and find the pokedex JSON."""
    names = zf.namelist()
    print("Archive contents:")
    for name in names:
        print(f"  {name} ({zf.getinfo(name).file_size} bytes)")
    pokedex_name = next(
        (n for n in names if n.endswith(".json") and "pokedex" in n), None
    )
    if not pokedex_name:
        raise SystemExit(
            f"FATAL: no pokedex JSON found in archive (found: {names})"
        )
    return pokedex_name


def write_schema_inspection(source_file: str, records: list[dict]) -> None:
    """Record the detected schema before any parse decisions are made."""
    detected = sorted(set().union(*(set(r.keys()) for r in records)))
    inspection = {
        "source_file": source_file,
        "record_count": len(records),
        "detected_fields": detected,
        "required_fields_present": {f: all(f in r for r in records) for f in REQUIRED_FIELDS},
    }
    SCHEMA_INSPECTION.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_INSPECTION.write_text(
        json.dumps(inspection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SCHEMA_INSPECTION}")
    print(
        f"  record_count={len(records)} "
        f"required_fields_present={inspection['required_fields_present']}"
    )


def load_raw_pokedex() -> list[dict]:
    """Return the full pokedex records, keying idempotence on the raw cache.

    If data/raw/complete_pokedex.json already exists it is reused — the dev
    subset in corpus.jsonl is a derived slice, never a re-download.
    """
    if RAW_POKEDEX.exists():
        print(f"Using cached raw dataset: {RAW_POKEDEX}")
        with open(RAW_POKEDEX, encoding="utf-8") as f:
            records = json.load(f)
        if not SCHEMA_INSPECTION.exists():
            write_schema_inspection(RAW_POKEDEX.name, records)
        return records

    zf = download_archive()
    pokedex_name = inspect_archive(zf)
    records = json.loads(zf.read(pokedex_name))
    if not isinstance(records, list):
        raise SystemExit(f"FATAL: expected a JSON list of records, got {type(records)}")

    write_schema_inspection(pokedex_name, records)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_POKEDEX.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"Persisted FULL raw dataset ({len(records)} records) to {RAW_POKEDEX}")
    return records


def parse_record(record: dict) -> dict:
    """Defensively map a raw Pokémon record into a normalized dict.

    Required minimum {id, name, types, stats, damage_taken}; every other field
    uses .get() fallbacks. If a required field is missing we fail loudly,
    listing the found keys — never a silently wrong parse.
    """
    found = set(record.keys())
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise SystemExit(
            "FATAL: record is missing required fields "
            f"{missing}. Found keys: {sorted(found)}"
        )

    stats = record.get("stats") or {}
    damage_taken = record.get("damage_taken") or {}

    return {
        "id": int(record["id"]),
        "name": str(record["name"]),
        "category": str(record.get("category") or ""),
        "generation": str(record.get("generation") or ""),
        "is_legendary": bool(record.get("is_legendary", False)),
        "is_mythical": bool(record.get("is_mythical", False)),
        "types": [str(t) for t in (record.get("types") or [])],
        "height": record.get("height"),
        "weight": record.get("weight"),
        "habitat": str(record.get("habitat") or ""),
        "evolves_from": str(record.get("evolves_from") or ""),
        "abilities": [str(a) for a in (record.get("abilities") or [])],
        "stats": {k: stats.get(k) for k in STAT_KEYS},
        "damage_taken": {k: damage_taken.get(k) for k in DAMAGE_TAKEN_KEYS},
        "description": str(record.get("description") or ""),
    }


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{value}{suffix}"


def build_passage(p: dict) -> str:
    """Build the readable structured passage for one Pokémon.

    Contains: name, category, generation, is_legendary/is_mythical, types,
    height (m), weight (kg), habitat, evolves_from, abilities (incl.
    "(Hidden)"), stats, and the full 18-key damage_taken table.
    """
    lines = [
        f"Name: {p['name']}",
        f"Category: {p['category'] or 'unknown'}",
        f"Generation: {p['generation'] or 'unknown'}",
        f"Legendary: {'Yes' if p['is_legendary'] else 'No'}",
        f"Mythical: {'Yes' if p['is_mythical'] else 'No'}",
        f"Types: {', '.join(p['types']) or 'unknown'}",
        f"Height: {_fmt(p['height'], ' m')}",
        f"Weight: {_fmt(p['weight'], ' kg')}",
        f"Habitat: {p['habitat'] or 'unknown'}",
        f"Evolves from: {p['evolves_from'] or 'None'}",
        f"Abilities: {', '.join(p['abilities']) if p['abilities'] else 'unknown'}",
    ]
    stats = p["stats"]
    stats_str = ", ".join(
        f"{k.replace('_', ' ')} {v}" for k, v in stats.items() if v is not None
    )
    lines.append(f"Stats: {stats_str or 'unknown'}")
    dt = p["damage_taken"]
    dt_str = ", ".join(f"{k} {v}" for k, v in dt.items() if v is not None)
    lines.append(f"damage_taken: {dt_str or 'unknown'}")
    if p["description"]:
        lines.append(f"Description: {p['description']}")
    return "\n".join(lines)


def build_corpus(records: list[dict], limit: int | None) -> list[dict]:
    """Select records (first N by id) and build the corpus rows."""
    records = sorted(records, key=lambda r: r["id"])
    selected = records if limit is None else records[:limit]
    return [{"id": p["id"], "passage": build_passage(p)} for p in (parse_record(r) for r in selected)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build data/corpus.jsonl from the Pokémon dataset (elroytan/pokemondata)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full", action="store_true", help="Write all 1,025 records (default is the 50-record dev subset)"
    )
    group.add_argument(
        "--limit", type=int, default=50, help="Number of records to write (default: 50, first N by id)"
    )
    args = parser.parse_args(argv)

    limit = None if args.full else args.limit

    records = load_raw_pokedex()
    rows = build_corpus(records, limit)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"Wrote {len(rows)} passages to {CORPUS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

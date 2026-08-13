import argparse
import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "patelris/pokemon-dataset-with-stats-and-types"
)

# Browser-like UA so Kaggle's static file endpoint serves the zip (no auth).
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATASET_PATH", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
POKEMON_CSV = RAW_DIR / "pokemon_complete.csv"
TYPES_CSV = RAW_DIR / "pokemon_types.csv"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"

# The 18 canonical types, in the order used for the "Type effectiveness"
# rendering (standard gen-1 ordering).
TYPE_ORDER = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "steel", "dark", "fairy",
]

STAT_KEYS = [
    "hp", "attack", "defense", "sp_attack", "sp_defense", "speed",
    "base_stat_total",
]


def validate_archive_body(content, content_type=""):
    """Validate a downloaded body is a ZIP archive before we try to open it.

    Kaggle's anonymous download endpoint is now bot-blocked and returns an
    HTTP 200 HTML page (Google reCAPTCHA) instead of the zip, which would
    otherwise crash later with an unhelpful BadZipFile. Raise SystemExit with
    a clear, actionable FATAL message instead. Returns None on success.
    """
    if not content.startswith(b"PK"):
        lower = content.lower()
        if b"recaptcha" in lower:
            raise SystemExit(
                "FATAL: anonymous Kaggle dataset downloads are currently blocked "
                "by a bot check (reCAPTCHA page returned instead of the archive). "
                "Use the bundled raw CSVs already present in data/raw/ (they ship "
                "with the repo - see docs/setup.md), or set KAGGLE_USERNAME and "
                "KAGGLE_KEY env vars and retry."
            )
        raise SystemExit(
            f"FATAL: unexpected response type (got {content_type or 'unknown'}, "
            f"{len(content)} bytes); expected a ZIP archive."
        )


def download_archive(url=KAGGLE_DATASET_URL):
    # Must always use GET: the Kaggle API endpoint returns 404 for HEAD
    # requests, and the signed GCS URL is never hardcoded - always go through it.
    print(f"Downloading archive from {url} ...")
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, allow_redirects=True,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"FATAL: archive download failed: {exc}") from exc
    print(f"  HTTP {resp.status_code}, {len(resp.content)} bytes")
    validate_archive_body(resp.content, resp.headers.get("content-type", ""))
    return zipfile.ZipFile(io.BytesIO(resp.content))


def extract_raw_csvs():
    # Idempotent on the raw cache: corpus.jsonl is a derived slice of it and is
    # never a re-download, so the raw CSVs are the deterministic source.
    if POKEMON_CSV.exists() and TYPES_CSV.exists():
        print(f"Using cached raw CSVs: {POKEMON_CSV}, {TYPES_CSV}")
        return

    zf = download_archive()
    names = zf.namelist()
    print("Archive contents:")
    for name in names:
        print(f"  {name} ({zf.getinfo(name).file_size} bytes)")

    wanted = {
        "pokemon_complete.csv": POKEMON_CSV,
        "pokemon_types.csv": TYPES_CSV,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        base = Path(name).name
        if base in wanted:
            RAW_DIR.joinpath(base).write_bytes(zf.read(name))
            print(f"Extracted {name} -> {RAW_DIR / base}")

    missing = [base for base in wanted if not wanted[base].exists()]
    if missing:
        raise SystemExit(
            f"FATAL: archive missing expected CSVs: {missing} (found: {names})"
        )


def to_int(value):
    value = (value or "").strip()
    return int(value) if value else None


def to_float(value):
    value = (value or "").strip()
    return float(value) if value else None


def to_bool(value):
    return (value or "").strip().lower() == "true"


def to_list(value):
    value = (value or "").strip()
    return [item.strip() for item in value.split("|") if item.strip()] if value else []


def to_str_or_none(value):
    value = (value or "").strip()
    return value if value else None


def parse_row(row):
    # CSV cells are all strings; empty cells -> null / empty list. Numeric
    # fields are parsed to int/float and never left as strings.
    types = [row["type_1"].strip()]
    if row["type_2"].strip():
        types.append(row["type_2"].strip())

    return {
        "id": int(row["pokedex_number"]),
        "name": row["name"].strip(),
        "types": types,
        "generation": row["generation"].strip(),
        "stats": {k: to_int(row[k]) for k in STAT_KEYS},
        "height_m": to_float(row["height_m"]),
        "weight_kg": to_float(row["weight_kg"]),
        "abilities": to_list(row["abilities"]),
        "hidden_ability": to_str_or_none(row["hidden_ability"]),
        "egg_groups": to_list(row["egg_groups"]),
        "color": to_str_or_none(row["color"]),
        "shape": to_str_or_none(row["shape"]),
        "habitat": to_str_or_none(row["habitat"]),
        "growth_rate": to_str_or_none(row["growth_rate"]),
        "capture_rate": to_int(row["capture_rate"]),
        "base_happiness": to_int(row["base_happiness"]),
        "base_experience": to_int(row["base_experience"]),
        "genus": to_str_or_none(row["genus"]),
        "is_legendary": to_bool(row["is_legendary"]),
        "is_mythical": to_bool(row["is_mythical"]),
        "is_baby": to_bool(row["is_baby"]),
        "evolution_chain_id": to_int(row["evolution_chain_id"]),
        "flavor_text": row["flavor_text"].strip(),
        "sprite_url": to_str_or_none(row["sprite_url"]),
    }


def load_raw_rows():
    extract_raw_csvs()
    with open(POKEMON_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_corpus(rows, limit):
    parsed = [parse_row(r) for r in rows]
    parsed.sort(key=lambda r: r["id"])
    if limit is not None:
        parsed = parsed[:limit]
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build data/corpus.jsonl from the Pokémon CSV dataset "
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
    corpus = build_corpus(rows, args.limit)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in corpus
        )
    print(f"Wrote {len(corpus)} records to {CORPUS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

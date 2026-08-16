import csv

from src.data.download import POKEMON_CSV, extract_raw_csvs

STAT_KEYS = [
    "hp",
    "attack",
    "defense",
    "sp_attack",
    "sp_defense",
    "speed",
    "base_stat_total",
]


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


# Declarative column -> key mapping; CSV cells are all strings, empty cells
# become null / empty list, and numeric fields are never left as strings.
FIELD_SPEC = [
    ("id", "pokedex_number", lambda v: int(v)),
    ("name", "name", str.strip),
    ("generation", "generation", str.strip),
    ("height_m", "height_m", to_float),
    ("weight_kg", "weight_kg", to_float),
    ("abilities", "abilities", to_list),
    ("hidden_ability", "hidden_ability", to_str_or_none),
    ("egg_groups", "egg_groups", to_list),
    ("color", "color", to_str_or_none),
    ("shape", "shape", to_str_or_none),
    ("habitat", "habitat", to_str_or_none),
    ("growth_rate", "growth_rate", to_str_or_none),
    ("capture_rate", "capture_rate", to_int),
    ("base_happiness", "base_happiness", to_int),
    ("base_experience", "base_experience", to_int),
    ("genus", "genus", to_str_or_none),
    ("is_legendary", "is_legendary", to_bool),
    ("is_mythical", "is_mythical", to_bool),
    ("is_baby", "is_baby", to_bool),
    ("evolution_chain_id", "evolution_chain_id", to_int),
    ("flavor_text", "flavor_text", str.strip),
    ("sprite_url", "sprite_url", to_str_or_none),
]


def parse_row(row):
    record = {key: convert(row[column]) for key, column, convert in FIELD_SPEC}
    types = [row["type_1"].strip()]
    if row["type_2"].strip():
        types.append(row["type_2"].strip())
    record["types"] = types
    record["stats"] = {k: to_int(row[k]) for k in STAT_KEYS}
    return record


def load_raw_rows():
    extract_raw_csvs()
    with open(POKEMON_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

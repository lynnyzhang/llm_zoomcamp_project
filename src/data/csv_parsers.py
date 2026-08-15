import csv

from src.data.download import POKEMON_CSV, extract_raw_csvs

STAT_KEYS = [
    "hp", "attack", "defense", "sp_attack", "sp_defense", "speed",
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

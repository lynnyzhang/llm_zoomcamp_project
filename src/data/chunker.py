import csv
import json
from collections import defaultdict
from pathlib import Path

# One Pokémon per document by design (content is short - a single labeled
# record), so no real chunk splitting happens. This constant is kept only as
# documentation of the intended token budget for the flat search_text rendering.
CHARS_PER_TOKEN = 4

# The 18 canonical attacker types, in the order used for the
# "Type effectiveness" rendering (standard gen-1 ordering).
TYPE_ORDER = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "steel", "dark", "fairy",
]

# Canonical Pokédex range; ids above this are alternate forms (Deoxys-Attack,
# Rotom-Heat, ...) which never participate in evolution linkage.
CANONICAL_MAX_ID = 1025

# Curated evolution overrides for chains where dex-id order != evolution
# order: baby Pokémon introduced after their parents, branched chains, and
# cross-generation evolutions. Keyed by canonical corpus name →
# (evolves_from, evolves_into). The dex-order fallback below only handles
# linear chains (where dex order == evolution order).
EVOLUTION_OVERRIDES = {
    # Baby chains (baby dex id > parent dex id)
    "Pichu": (None, ["Pikachu"]),
    "Pikachu": ("Pichu", ["Raichu"]),
    "Raichu": ("Pikachu", []),
    "Cleffa": (None, ["Clefairy"]),
    "Clefairy": ("Cleffa", ["Clefable"]),
    "Clefable": ("Clefairy", []),
    "Igglybuff": (None, ["Jigglypuff"]),
    "Jigglypuff": ("Igglybuff", ["Wigglytuff"]),
    "Wigglytuff": ("Jigglypuff", []),
    "Azurill": (None, ["Marill"]),
    "Marill": ("Azurill", ["Azumarill"]),
    "Azumarill": ("Marill", []),
    "Elekid": (None, ["Electabuzz"]),
    "Electabuzz": ("Elekid", ["Electivire"]),
    "Electivire": ("Electabuzz", []),
    "Magby": (None, ["Magmar"]),
    "Magmar": ("Magby", ["Magmortar"]),
    "Magmortar": ("Magmar", []),
    "Smoochum": (None, ["Jynx"]),
    "Jynx": ("Smoochum", []),
    "Tyrogue": (None, ["Hitmonlee", "Hitmonchan", "Hitmontop"]),
    "Hitmonlee": ("Tyrogue", []),
    "Hitmonchan": ("Tyrogue", []),
    "Hitmontop": ("Tyrogue", []),
    "Wynaut": (None, ["Wobbuffet"]),
    "Wobbuffet": ("Wynaut", []),
    "Budew": (None, ["Roselia"]),
    "Roselia": ("Budew", ["Roserade"]),
    "Roserade": ("Roselia", []),
    "Chingling": (None, ["Chimecho"]),
    "Chimecho": ("Chingling", []),
    "Bonsly": (None, ["Sudowoodo"]),
    "Sudowoodo": ("Bonsly", []),
    "Mime-Jr": (None, ["Mr-Mime"]),
    "Mr-Mime": ("Mime-Jr", ["Mr-Rime"]),
    "Mr-Rime": ("Mr-Mime", []),
    "Happiny": (None, ["Chansey"]),
    "Chansey": ("Happiny", ["Blissey"]),
    "Blissey": ("Chansey", []),
    "Munchlax": (None, ["Snorlax"]),
    "Snorlax": ("Munchlax", []),
    "Mantyke": (None, ["Mantine"]),
    "Mantine": ("Mantyke", []),
    # Branched chains (dex order breaks the linear chain)
    "Oddish": (None, ["Gloom"]),
    "Gloom": ("Oddish", ["Vileplume", "Bellossom"]),
    "Vileplume": ("Gloom", []),
    "Bellossom": ("Gloom", []),
    "Poliwag": (None, ["Poliwhirl"]),
    "Poliwhirl": ("Poliwag", ["Poliwrath", "Politoed"]),
    "Poliwrath": ("Poliwhirl", []),
    "Politoed": ("Poliwhirl", []),
    "Wurmple": (None, ["Silcoon", "Cascoon"]),
    "Silcoon": ("Wurmple", ["Beautifly"]),
    "Cascoon": ("Wurmple", ["Dustox"]),
    "Beautifly": ("Silcoon", []),
    "Dustox": ("Cascoon", []),
    "Ralts": (None, ["Kirlia"]),
    "Kirlia": ("Ralts", ["Gardevoir", "Gallade"]),
    "Gardevoir": ("Kirlia", []),
    "Gallade": ("Kirlia", []),
    "Eevee": (None, ["Vaporeon", "Jolteon", "Flareon", "Espeon", "Umbreon", "Leafeon", "Glaceon", "Sylveon"]),
    "Vaporeon": ("Eevee", []),
    "Jolteon": ("Eevee", []),
    "Flareon": ("Eevee", []),
    "Espeon": ("Eevee", []),
    "Umbreon": ("Eevee", []),
    "Leafeon": ("Eevee", []),
    "Glaceon": ("Eevee", []),
    "Sylveon": ("Eevee", []),
    "Slowpoke": (None, ["Slowbro", "Slowking"]),
    "Slowbro": ("Slowpoke", []),
    "Slowking": ("Slowpoke", []),
    "Snorunt": (None, ["Glalie", "Froslass"]),
    "Glalie": ("Snorunt", []),
    "Froslass": ("Snorunt", []),
    "Clamperl": (None, ["Huntail", "Gorebyss"]),
    "Huntail": ("Clamperl", []),
    "Gorebyss": ("Clamperl", []),
    "Nincada": (None, ["Ninjask", "Shedinja"]),
    "Ninjask": ("Nincada", []),
    "Shedinja": ("Nincada", []),
    "Scyther": (None, ["Scizor", "Kleavor"]),
    "Scizor": ("Scyther", []),
    "Kleavor": ("Scyther", []),
    "Meowth": (None, ["Persian", "Perrserker"]),
    "Persian": ("Meowth", []),
    "Perrserker": ("Meowth", []),
    "Sneasel": (None, ["Weavile", "Sneasler"]),
    "Weavile": ("Sneasel", []),
    "Sneasler": ("Sneasel", []),
    "Yamask": (None, ["Cofagrigus", "Runerigus"]),
    "Cofagrigus": ("Yamask", []),
    "Runerigus": ("Yamask", []),
    "Wooper": (None, ["Quagsire", "Clodsire"]),
    "Quagsire": ("Wooper", []),
    "Clodsire": ("Wooper", []),
    "Burmy": (None, ["Wormadam-Plant", "Mothim"]),
    "Wormadam-Plant": ("Burmy", []),
    "Mothim": ("Burmy", []),
    "Cosmog": (None, ["Cosmoem"]),
    "Cosmoem": ("Cosmog", ["Solgaleo", "Lunala"]),
    "Solgaleo": ("Cosmoem", []),
    "Lunala": ("Cosmoem", []),
    "Charcadet": (None, ["Armarouge", "Ceruledge"]),
    "Armarouge": ("Charcadet", []),
    "Ceruledge": ("Charcadet", []),
    "Applin": (None, ["Flapple", "Appletun", "Dipplin"]),
    "Flapple": ("Applin", []),
    "Appletun": ("Applin", []),
    "Dipplin": ("Applin", ["Hydrapple"]),
    "Hydrapple": ("Dipplin", []),
    # Bred, not evolved (dataset groups them in one chain)
    "Phione": (None, []),
    "Manaphy": (None, []),
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PATH = _PROJECT_ROOT / "data" / "corpus.jsonl"
_TYPES_CSV_PATH = _PROJECT_ROOT / "data" / "raw" / "pokemon_types.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "chunks" / "documents.jsonl"

_TYPE_CHART_FIELDS = [
    "double_damage_to",
    "half_damage_to",
    "no_damage_to",
    "double_damage_from",
    "half_damage_from",
    "no_damage_from",
]


def load_type_chart(path):
    """Load pokemon_types.csv into {lowercase_type: {field: [lowercase, ...]}}."""
    chart = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["type"].strip().lower()
            chart[t] = {
                field: [
                    item.strip().lower()
                    for item in row[field].split("|")
                    if item.strip()
                ]
                for field in _TYPE_CHART_FIELDS
            }
    return chart


def _type_chart_doc(chart, t):
    """Build one Part B type-chart document from a chart row key `t`."""
    title = t.capitalize()

    def render_list(value):
        return ", ".join(v.capitalize() for v in value) if value else "[none]"

    def render_dir(field):
        return render_list(chart[t][field])

    search_text = (
        f"Type chart: {title}. {title} moves deal 2x damage to "
        f"{render_dir('double_damage_to')}; 0.5x damage to "
        f"{render_dir('half_damage_to')}; no effect on "
        f"{render_dir('no_damage_to')}. {title} Pokémon take 2x damage from "
        f"{render_dir('double_damage_from')}; 0.5x from "
        f"{render_dir('half_damage_from')}; no damage from "
        f"{render_dir('no_damage_from')}."
    )

    doc = {"id": f"type_{t}", "kind": "type_chart", "type": title}
    for field in _TYPE_CHART_FIELDS:
        doc[field] = [v.capitalize() for v in chart[t][field]]
    doc["search_text"] = search_text
    return doc


def build_evolution_map(corpus):
    """Map evolution_chain_id -> sorted list of canonical member records.

    Only canonical dex ids (<= CANONICAL_MAX_ID) are considered members, and
    members are ordered by dex id ascending.
    """
    chains = defaultdict(list)
    for record in corpus:
        if record["id"] > CANONICAL_MAX_ID:
            continue
        chains[record["evolution_chain_id"]].append(record)
    for members in chains.values():
        members.sort(key=lambda r: r["id"])
    return chains


def _evolution_linkage(record, chain):
    """Return (evolves_from, evolves_into) for a record within its chain.

    Curated overrides first (baby/branched chains where dex order !=
    evolution order); the dex-order heuristic is the fallback.
    """
    if record["id"] > CANONICAL_MAX_ID or not chain:
        return None, []
    override = EVOLUTION_OVERRIDES.get(record["name"])
    if override is not None:
        return override
    names = [r["name"] for r in chain]
    idx = next(
        (i for i, r in enumerate(chain) if r["id"] == record["id"]), None
    )
    if idx is None:
        return None, []
    evolves_from = names[idx - 1] if idx > 0 else None
    evolves_into = [names[idx + 1]] if idx + 1 < len(names) else []
    return evolves_from, evolves_into


def _chain_render_order(chain):
    """Evolution-order names for a chain, using the curated overrides when
    the dex-order heuristic would misorder it (baby chains); otherwise the
    dex order is already a valid evolution order."""
    names = [r["name"] for r in chain]
    if not any(name in EVOLUTION_OVERRIDES for name in names):
        return names
    by_name = {r["name"] for r in chain}
    ordered = []
    queue = [
        name for name in names
        if EVOLUTION_OVERRIDES.get(name, (None, []))[0] is None
    ]
    while queue:
        name = queue.pop(0)
        if name in ordered:
            continue
        ordered.append(name)
        queue.extend(
            child for child in EVOLUTION_OVERRIDES.get(name, (None, []))[1]
            if child in by_name
        )
    ordered.extend(n for n in names if n not in ordered)
    return ordered


def _chain_render(record, chain):
    """Render the 'Evolution chain' line of search_text for a record."""
    if record["id"] > CANONICAL_MAX_ID or not chain:
        return "Evolution chain: none"
    names = _chain_render_order(chain)
    if len(names) < 2:
        return f"Evolution chain: {record['evolution_chain_id']} (single member)"
    path = " -> ".join(names)
    return (
        f"Evolution chain: {record['evolution_chain_id']} ({path}) [derived]"
    )


def type_effectiveness(record, chart):
    """Multiplier each attacker type deals to this Pokémon.

    For each of the 18 attacker types: product over the Pokémon's types t of
    2.0 if attacker in chart[t].double_damage_from, 0.5 if in half_damage_from,
    0.0 if in no_damage_from, else 1.0.
    """
    result = {}
    for attacker in TYPE_ORDER:
        mult = 1.0
        for t in record["types"]:
            row = chart[t.lower()]
            if attacker in row["double_damage_from"]:
                mult *= 2.0
            elif attacker in row["half_damage_from"]:
                mult *= 0.5
            elif attacker in row["no_damage_from"]:
                mult *= 0.0
        result[attacker] = mult
    return result


def _yes_no(value):
    return "yes" if value else "no"


def build_search_text(record, chart, chain):
    """The flat labeled rendering used for embedding + keyword search."""
    stats = record["stats"]
    abilities = ", ".join(record["abilities"]) if record["abilities"] else "none"
    if record["hidden_ability"]:
        abilities += f"; hidden: {record['hidden_ability']}"

    eff = type_effectiveness(record, chart)
    eff_str = ", ".join(f"{k} {v}" for k, v in eff.items())

    lines = [
        f"Pokémon: {record['name']} (#{record['id']})",
        f"Genus: {record['genus']}",
        f"Types: {', '.join(record['types'])}",
        f"Generation: {record['generation']}",
        f"Height: {record['height_m']} m",
        f"Weight: {record['weight_kg']} kg",
        f"Abilities: {abilities}",
        f"Egg groups: {', '.join(record['egg_groups'])}",
        f"Color: {record['color']}",
        f"Shape: {record['shape']}",
    ]
    if record["habitat"]:
        lines.append(f"Habitat: {record['habitat']}")
    lines.append(f"Growth rate: {record['growth_rate']}")
    lines.append(f"Capture rate: {record['capture_rate']}")
    lines.append(f"Base happiness: {record['base_happiness']}")
    if record["base_experience"] is not None:
        lines.append(f"Base experience: {record['base_experience']}")
    lines.append(
        f"Stats: hp {stats['hp']}, attack {stats['attack']}, "
        f"defense {stats['defense']}, sp. attack {stats['sp_attack']}, "
        f"sp. defense {stats['sp_defense']}, speed {stats['speed']}, "
        f"total {stats['base_stat_total']}"
    )
    lines.append(
        f"Flags: legendary {_yes_no(record['is_legendary'])}, "
        f"mythical {_yes_no(record['is_mythical'])}, "
        f"baby {_yes_no(record['is_baby'])}"
    )
    lines.append(_chain_render(record, chain))
    lines.append(f"Type effectiveness: {eff_str}")
    lines.append(f"Flavor text: {record['flavor_text']}")
    return "\n".join(lines)


def build_pokemon_doc(record, chart, chain):
    evolves_from, evolves_into = _evolution_linkage(record, chain)
    doc = dict(record)
    doc["evolves_from"] = evolves_from
    doc["evolves_into"] = evolves_into
    doc["type_effectiveness"] = type_effectiveness(record, chart)
    doc["search_text"] = build_search_text(record, chart, chain)
    return doc


def main():
    if not _CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Missing corpus at {_CORPUS_PATH}. "
            "Run `uv run python -m src.data.ingest` first."
        )
    if not _TYPES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing type chart at {_TYPES_CSV_PATH}. "
            "Run `uv run python -m src.data.ingest` first."
        )

    with open(_CORPUS_PATH, encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    chart = load_type_chart(_TYPES_CSV_PATH)
    chains = build_evolution_map(corpus)

    docs = [
        build_pokemon_doc(record, chart, chains.get(record["evolution_chain_id"]))
        for record in corpus
    ]
    # Part B: 18 type-chart docs appended after the 1,350 Pokémon docs,
    # in the order the rows appear in pokemon_types.csv (dict preserves order).
    for t in chart:
        docs.append(_type_chart_doc(chart, t))

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs)

    print(f"Processed {_CORPUS_PATH}")
    print(f"Saved {len(docs)} docs to {_OUTPUT_PATH}")

    # Quick validation
    with open(_OUTPUT_PATH, encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"Sample document keys: {list(first.keys())}")


if __name__ == "__main__":
    main()

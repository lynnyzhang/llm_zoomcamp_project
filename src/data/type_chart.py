import csv

# The 18 canonical attacker types, in the order used for the
# "Type effectiveness" rendering (standard gen-1 ordering).
TYPE_ORDER = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "steel", "dark", "fairy",
]

TYPE_CHART_FIELDS = [
    "double_damage_to",
    "half_damage_to",
    "no_damage_to",
    "double_damage_from",
    "half_damage_from",
    "no_damage_from",
]


class TypeChart:
    """Pokémon type matchups, loaded from pokemon_types.csv.

    Maps lowercase type -> {field: [lowercase, ...]}. Renders the type-chart
    documents and the per-Pokémon effectiveness table.
    """

    def __init__(self, chart):
        self.chart = chart

    @classmethod
    def load(cls, path):
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
                    for field in TYPE_CHART_FIELDS
                }
        return cls(chart)

    def doc(self, type_name):
        """Build one Part B type-chart document for type `type_name`."""
        title = type_name.capitalize()

        def render_list(value):
            return ", ".join(v.capitalize() for v in value) if value else "[none]"

        def render_dir(field):
            return render_list(self.chart[type_name][field])

        search_text = (
            f"Type chart: {title}. {title} moves deal 2x damage to "
            f"{render_dir('double_damage_to')}; 0.5x damage to "
            f"{render_dir('half_damage_to')}; no effect on "
            f"{render_dir('no_damage_to')}. {title} Pokémon take 2x damage from "
            f"{render_dir('double_damage_from')}; 0.5x from "
            f"{render_dir('half_damage_from')}; no damage from "
            f"{render_dir('no_damage_from')}."
        )

        doc = {"id": f"type_{type_name}", "kind": "type_chart", "type": title}
        for field in TYPE_CHART_FIELDS:
            doc[field] = [v.capitalize() for v in self.chart[type_name][field]]
        doc["search_text"] = search_text
        return doc

    def effectiveness(self, record):
        """Multiplier each attacker type deals to this Pokémon.

        For each of the 18 attacker types: product over the Pokémon's types t
        of 2.0 if attacker in chart[t].double_damage_from, 0.5 if in
        half_damage_from, 0.0 if in no_damage_from, else 1.0.
        """
        result = {}
        for attacker in TYPE_ORDER:
            mult = 1.0
            for t in record["types"]:
                row = self.chart[t.lower()]
                if attacker in row["double_damage_from"]:
                    mult *= 2.0
                elif attacker in row["half_damage_from"]:
                    mult *= 0.5
                elif attacker in row["no_damage_from"]:
                    mult *= 0.0
            result[attacker] = mult
        return result

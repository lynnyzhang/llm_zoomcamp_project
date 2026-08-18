from src.data.evolution import EvolutionChain
from src.data.type_chart import TypeChart


class PokemonDocBuilder:
    """Builds the flat labeled search_text and the final Pokémon document."""

    @staticmethod
    def yes_no(value):
        return "yes" if value else "no"

    def build_search_text(self, record, chart, chain):
        """The flat labeled rendering used for embedding + keyword search."""
        stats = record["stats"]
        abilities = ", ".join(record["abilities"]) if record["abilities"] else "none"
        if record["hidden_ability"]:
            abilities += f"; hidden: {record['hidden_ability']}"

        eff = chart.effectiveness(record)
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
            f"Flags: legendary {self.yes_no(record['is_legendary'])}, "
            f"mythical {self.yes_no(record['is_mythical'])}, "
            f"baby {self.yes_no(record['is_baby'])}"
        )
        lines.append(EvolutionChain.render(record, chain))
        lines.append(f"Type effectiveness: {eff_str}")
        lines.append(f"Flavor text: {record['flavor_text']}")
        return "\n".join(lines)

    def build(self, record, chart, chain):
        doc = dict(record)
        doc["search_text"] = self.build_search_text(record, chart, chain)
        return doc

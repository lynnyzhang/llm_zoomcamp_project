from collections import defaultdict

from src.data.evolution_overrides import EVOLUTION_OVERRIDES

# Canonical Pokédex range; ids above this are alternate forms (Deoxys-Attack,
# Rotom-Heat, ...) which never participate in evolution linkage.
CANONICAL_MAX_ID = 1025


class EvolutionChain:
    """Resolves evolution linkage and ordering for the Pokémon records.

    Curated overrides first (baby/branched chains where dex order != evolution
    order); the dex-order heuristic is the fallback.
    """

    @staticmethod
    def build_map(records):
        """Map evolution_chain_id -> sorted list of canonical member records.

        Only canonical dex ids (<= CANONICAL_MAX_ID) are considered members,
        and members are ordered by dex id ascending.
        """
        chains = defaultdict(list)
        for record in records:
            if record["id"] > CANONICAL_MAX_ID:
                continue
            chains[record["evolution_chain_id"]].append(record)
        for members in chains.values():
            members.sort(key=lambda r: r["id"])
        return chains

    @staticmethod
    def link(record, chain):
        """Return (evolves_from, evolves_into) for a record within its chain."""
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

    @staticmethod
    def render_order(chain):
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

    @staticmethod
    def render(record, chain):
        if record["id"] > CANONICAL_MAX_ID or not chain:
            return "Evolution chain: none"
        names = EvolutionChain.render_order(chain)
        if len(names) < 2:
            return f"Evolution chain: {record['evolution_chain_id']} (single member)"
        path = " -> ".join(names)
        return (
            f"Evolution chain: {record['evolution_chain_id']} ({path}) [derived]"
        )

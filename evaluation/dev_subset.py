import random

DEV_SUBSET_SIZE = 50
DEV_SUBSET_SEED = 42


def select_dev_subset(records, size=DEV_SUBSET_SIZE, seed=DEV_SUBSET_SEED):
    """Deterministic coverage-sampled subset: all 18 types, every generation,
    and legendary/mythical representation, then a generation-balanced fill.
    Same input + seed → same output."""
    records = sorted(records, key=lambda r: int(r["id"]))
    rng = random.Random(seed)

    def attrs(r):
        a = set(r.get("types") or [])
        a.add(("gen", str(r.get("generation") or "unknown")))
        if r.get("is_legendary"):
            a.add("legendary")
        if r.get("is_mythical"):
            a.add("mythical")
        return a

    selected, covered, remaining = [], set(), records[:]
    # Greedy set cover: repeatedly take the record adding the most new
    # coverage attributes (types/generation/rarity); seeded-random tie-break.
    while len(selected) < size and remaining:
        scored = [(len(attrs(r) - covered), r) for r in remaining]
        best = max(g for g, _ in scored)
        if best == 0:
            break
        candidates = [r for g, r in scored if g == best]
        pick = rng.choice(candidates)
        selected.append(pick)
        covered |= attrs(pick)
        remaining.remove(pick)
    # Stratified fill: round-robin across generations for balance.
    if len(selected) < size and remaining:
        by_gen = {}
        for r in remaining:
            by_gen.setdefault(str(r.get("generation") or "unknown"), []).append(r)
        i = 0
        while len(selected) < size and any(i < len(by_gen[g]) for g in by_gen):
            for g in sorted(by_gen):
                if i < len(by_gen[g]) and len(selected) < size:
                    selected.append(by_gen[g][i])
            i += 1
        if len(selected) < size:  # defensive; cannot happen with 1,350 documents
            selected.extend(remaining[: size - len(selected)])
    return selected


def coverage_summary(records):
    types, gens = set(), set()
    legendary = mythical = 0
    for r in records:
        types.update(r.get("types") or [])
        gens.add(str(r.get("generation") or "unknown"))
        legendary += 1 if r.get("is_legendary") else 0
        mythical += 1 if r.get("is_mythical") else 0
    return types, gens, legendary, mythical


def resolve_ids(records, limit, full, seed):
    if full:
        return sorted(records)
    return sorted(int(r["id"]) for r in select_dev_subset(list(records.values()), size=(limit if limit is not None else DEV_SUBSET_SIZE), seed=seed))

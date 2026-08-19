import json
import random
import time

from pydantic import BaseModel

from evaluation.notebooks.share.llm_calls import llm_structured


# --- Dev subset selection -------------------------------------------------

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
    return sorted(
        int(r["id"])
        for r in select_dev_subset(
            list(records.values()),
            size=(limit if limit is not None else DEV_SUBSET_SIZE),
            seed=seed,
        )
    )


# --- Question generation --------------------------------------------------

# PLAN DEVIATION (documented, 2026-08-07): the plan calls for max_workers 6,
# but the local llama-server degrades under concurrent load — measured: 6
# workers stalled the run entirely, 3+ tripled per-request latency, while 2
# workers sustained ~2x sequential throughput (~21s/record wall). 2 is the
# observed sweet spot for this endpoint (4 slots but CPU-bound); bump for
# faster/parallel servers.
MAX_WORKERS = 2

# Reference-style instructions: generate only questions; the record must
# contain the answer; use as few record words as possible so retrieval is not
# trivially exact-matchable.
DATA_GEN_INSTRUCTIONS = """
You emulate a Pokémon fan asking questions on the internet.
You are given one Pokédex record for a Pokémon.
Formulate 5 questions this fan might ask that are answered by this record
(stats, types, type effectiveness, abilities, evolution).

Rules:
- The record must contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the record; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the Pokémon, not about the record's formatting.

Respond with a single JSON object in exactly this format (no markdown, no
extra text):
{"questions": ["question 1", "question 2", "question 3", "question 4", "question 5"]}
""".strip()

# The model sometimes truncates the JSON mid-list — retries in
# generate_questions cover it; top up any shortfall with follow-up
# generations.
TARGET_QUESTIONS_PER_RECORD = 5
FILL_ATTEMPTS = 3

FILL_INSTRUCTIONS = """
You emulate a Pokémon fan asking questions on the internet.
You are given one Pokédex record for a Pokémon.
Formulate exactly {needed} additional natural questions this fan might ask
that are answered by this record (stats, types, type effectiveness, abilities,
evolution). Do not repeat questions already asked.

Rules:
- The record must contain the answer to each question.
- Use as few words as possible from the record; don't copy its phrasing.
- The questions should resemble how people actually ask things online.

Respond with a single JSON object in exactly this format (no markdown, no
extra text):
{"questions": ["question 1", "question 2", "..."]}
""".strip()


class Questions(BaseModel):
    questions: list[str]


def generate_questions(client, model, record, instructions=DATA_GEN_INSTRUCTIONS):
    user_prompt = json.dumps(record)
    # Single path: structured output via responses.parse(text_format=Questions).
    # The eval scripts use their own patch_openai_client for llama.cpp
    # (response_format via extra_body), passing through to the native SDK on
    # OpenAI. There is deliberately NO create()/JSON extraction fallback: if the
    # backend cannot honor the schema, the output is prose — not JSON — so there
    # is nothing to salvage; fail loudly. Retry with backoff covers transient
    # failures.
    for attempt in range(3):
        try:
            parsed, _ = llm_structured(
                client, instructions, user_prompt, Questions, model=model
            )
            if parsed is None:
                raise ValueError(
                    "structured output unsupported: server returned output_parsed=None"
                )
            return parsed.questions
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def generate_questions_for_record(pokemon_id, record, client, model, target_questions):
    seen = set()
    rows = []

    def add(questions):
        for q in questions:
            question = (q or "").strip()
            if question and question not in seen:
                seen.add(question)
                rows.append({"question": question, "document": pokemon_id})

    add(generate_questions(client, model, record))
    for _ in range(FILL_ATTEMPTS):
        if len(rows) >= target_questions:
            break
        needed = target_questions - len(rows)
        add(
            generate_questions(
                client,
                model,
                record,
                instructions=FILL_INSTRUCTIONS.replace("{needed}", str(needed)),
            )
        )
    return rows[:target_questions]

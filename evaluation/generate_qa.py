# LLM-generated Pokémon ground-truth set (evaluation/data/qa.jsonl): per-record
# natural-language QUESTIONS linked to the Pokédex document that contains the
# answer. Mirrors the course's data-generation pattern for grounded QA sets:
# only questions are LLM-generated; the ground-truth answer is the document
# itself, looked up by id at eval time — the LLM never writes answers.
#
# Records come from the INDEXED documents (data/chunks/documents.jsonl) — the
# same corpus the retrievers index — so questions are grounded in exactly what
# retrieval sees (search_text, type_effectiveness, evolution links), not in the
# raw CSVs.

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from evaluation.evaluation_utils import llm_structured, map_progress
from src.llm import LLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "chunks" / "documents.jsonl"
QA_FILE = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"

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
DEV_SUBSET_SIZE = 50
DEV_SUBSET_SEED = 42

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
    # The eval scripts use their own patch_openai_client (evaluation_utils)
    # for llama.cpp (response_format via extra_body), passing through to the
    # native SDK on OpenAI. There is deliberately NO
    # create()/JSON extraction fallback: if the backend cannot honor the
    # schema, the output is prose — not JSON — so there is nothing to
    # salvage; fail loudly. Retry with backoff covers transient failures.
    for attempt in range(3):
        try:
            parsed, _ = llm_structured(client, instructions, user_prompt, Questions, model=model)
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


def load_pokemon_documents():
    """data/chunks/documents.jsonl → {int id: Pokémon document}.

    The indexed documents are the ground truth for question generation (the
    course's pattern generates questions from the indexed corpus, not from raw
    data). Type-chart docs (kind == "type_chart", string ids) are excluded —
    QA is per Pokémon.
    """
    if not DOCUMENTS_FILE.exists():
        raise SystemExit(
            f"FATAL: indexed documents missing at {DOCUMENTS_FILE}. "
            "Run `uv run python -m src.data.chunker` first."
        )
    docs = {}
    with open(DOCUMENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("kind") == "type_chart":
                continue
            docs[doc["id"]] = doc
    return docs


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate LLM ground-truth questions into evaluation/data/qa.jsonl "
        "(default: deterministic coverage-sampled dev subset — 50 Pokémon × 5 questions; "
        "each row links a question to the indexed Pokédex document that contains its answer)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="All 1,350 Pokémon documents × N questions (MANUAL only — slow/costly)")
    group.add_argument("--limit", type=int, default=DEV_SUBSET_SIZE,
                       help=f"Coverage-sampled N Pokémon documents from the indexed corpus (default: {DEV_SUBSET_SIZE})")
    parser.add_argument("--questions", type=int, default=TARGET_QUESTIONS_PER_RECORD,
                        help="Target questions per record (default: 5; fewer = cheaper but weaker stats)")
    parser.add_argument("--seed", type=int, default=DEV_SUBSET_SEED,
                        help=f"Seed for the deterministic coverage sampler (default: {DEV_SUBSET_SEED})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip record ids already present in evaluation/data/qa.jsonl (safe re-run after a crash)")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")

    records = load_pokemon_documents()
    ids = resolve_ids(records, args.limit, args.full, args.seed)

    if args.resume and QA_FILE.exists():
        existing = set()
        with open(QA_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["document"])
        before = len(ids)
        ids = [i for i in ids if i not in existing]
        print(f"Resume: skipped {before - len(ids)} ids already in {QA_FILE}")

    print(f"Records to generate for: {len(ids)}")
    if ids:
        sel_records = [records[i] for i in ids]
        types, gens, legendary, mythical = coverage_summary(sel_records)
        print(f"  coverage: {len(types)}/18 types, {len(gens)} generations, "
              f"{legendary} legendary, {mythical} mythical (seed={args.seed})")

    client = LLMClient.get()
    # Bound the SDK's default 600s per-request timeout and disable its built-in
    # retries (the generate_questions loop below is the retry layer): a
    # black-holing endpoint must fail fast, not hang for minutes per attempt.
    # LLMClient.get() returns the LLMClient wrapper — configure the underlying
    # OpenAI client it lazily creates.
    client.client.timeout = 120.0
    client.client.max_retries = 0
    model = LLMClient.get_model()
    print(f"Model: {model} | expected questions: {len(ids) * args.questions}")

    all_rows = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = map_progress(
                pool,
                ids,
                lambda pokemon_id: generate_questions_for_record(
                    pokemon_id, records[pokemon_id], client, model, args.questions
                ),
            )
        for rows in results:
            all_rows.extend(rows)
    except Exception as exc:  # noqa: BLE001 — any worker/SDK failure → clean non-zero exit
        print(f"FATAL: QA generation failed: {exc}", file=sys.stderr)
        print(
            "No changes written to evaluation/data/qa.jsonl (written atomically only on success).",
            file=sys.stderr,
        )
        return 1

    all_rows.sort(key=lambda row: row["document"])
    QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_FILE, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows)
    print(f"Wrote {len(all_rows)} ground-truth questions to {QA_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

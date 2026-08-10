# LLM-generated Pokémon Q&A set (evaluation/data/qa.jsonl): per-record natural-language
# questions with answers grounded in the record.

import argparse
import json
import random
import re
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.llm import create_client, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_POKEDEX = DATA_DIR / "raw" / "complete_pokedex.json"
QA_FILE = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"

# PLAN DEVIATION (documented, 2026-08-07): the plan calls for max_workers 6,
# but the local llama-server degrades under concurrent load — measured: 6
# workers stalled the run entirely, 3+ tripled per-request latency, while 2
# workers sustained ~2x sequential throughput (~21s/record wall). 2 is the
# observed sweet spot for this endpoint (4 slots but CPU-bound); bump for
# faster/parallel servers.
MAX_WORKERS = 2

DATA_GEN_INSTRUCTIONS = """
You emulate a Pokémon fan asking questions on the internet.
Formulate 5 natural questions this fan might ask based on this Pokémon record
(stats, types, weaknesses, abilities, evolution). The record must contain the
answer. Not too formal, not too short, not too long.
For each question also provide the answer grounded in the record.

Respond with a single JSON object in exactly this format (no markdown, no
extra text):
{"qa_pairs": [{"question": "the question", "answer": "the answer"}]}
""".strip()

# The model sometimes truncates the JSON mid-list (salvaged by repair into a
# valid but short array) — top-up shortfalls with follow-up generations.
TARGET_PAIRS_PER_RECORD = 5
FILL_ATTEMPTS = 3
DEV_SUBSET_SIZE = 50
DEV_SUBSET_SEED = 42

FILL_INSTRUCTIONS = """
You emulate a Pokémon fan asking questions on the internet.
Formulate exactly {needed} additional natural questions this fan might ask
based on this Pokémon record (stats, types, weaknesses, abilities, evolution),
along with answers grounded in the record. Do not repeat questions already
asked. Not too formal, not too short, not too long.

Respond with a single JSON object in exactly this format (no markdown, no
extra text):
{"qa_pairs": [{"question": "the question", "answer": "the answer"}]}
""".strip()


class QAPair(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: list[QAPair]


def patch_openai_client(client):
    # responses.parse sends a "text" field in the request body that llama.cpp
    # rejects; inject the pydantic json_schema via extra_body.response_format
    # instead (falls back to native parse when no text_format is given).
    _original_parse = client.responses.parse

    def _patched_responses_parse(self, model, input, **kwargs):
        pydantic_model = kwargs.pop("text_format", None)
        if pydantic_model is None:
            return _original_parse(model=model, input=input, **kwargs)

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": pydantic_model.__name__,
                "strict": True,
                "schema": pydantic_model.model_json_schema(),
            },
        }
        extra_body = kwargs.pop("extra_body", {})
        extra_body["response_format"] = json_schema
        return _original_parse(model=model, input=input, extra_body=extra_body, **kwargs)

    client.responses.parse = types.MethodType(_patched_responses_parse, client.responses)
    return client


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    body = text[start:]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = _repair_json(body)
    if not isinstance(data, dict):
        raise TypeError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _repair_json(text):
    stripped = text.rstrip()
    variants = [stripped]
    if not stripped.endswith("}"):
        variants.append(stripped + "}")
    end = stripped.rfind("}")
    if end > 0:
        sliced = stripped[: end + 1]
        variants.append(sliced)
        if not sliced.endswith("}"):
            variants.append(sliced + "}")
    for variant in variants:
        try:
            data = json.loads(re.sub(r",\s*([\]}])", r"\1", variant))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"could not repair model JSON output: {text[:80]}...")


def _try_parse(client, model, messages):
    try:
        parsed = client.responses.parse(model=model, input=messages, text_format=QAResponse).output_parsed
    except Exception:  # noqa: BLE001 — servers that reject parse fall back to create
        return None
    return parsed.qa_pairs if parsed is not None else None


def _generate_qa_pairs(client, model, record, use_parse, instructions=DATA_GEN_INSTRUCTIONS):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": json.dumps(record)},
    ]
    if use_parse:
        pairs = _try_parse(client, model, messages)
        if pairs is not None:
            return pairs
    # Servers that ignore response_format return prose (output_parsed=None);
    # the JSON contract in the instructions makes the completion parseable.
    response = client.responses.create(model=model, input=messages)
    return QAResponse.model_validate(_extract_json(response.output_text)).qa_pairs


def supports_structured_output(client, model):
    # Local servers (e.g. llama.cpp) accept the request but return plain text
    # (output_parsed=None), silently wasting a generation — probe once and skip.
    try:
        probe = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": 'Reply with JSON only: {"qa_pairs": []}'}],
            text_format=QAResponse,
            max_output_tokens=64,
        )
        return probe.output_parsed is not None
    except Exception:  # noqa: BLE001 — any rejection means JSON fallback mode
        return False


def llm_structured_retry(
    client,
    model,
    record,
    use_parse,
    max_retries=3,
    instructions=DATA_GEN_INSTRUCTIONS,
):
    for attempt in range(max_retries):
        try:
            return _generate_qa_pairs(client, model, record, use_parse, instructions)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


def generate_for_record(pokemon_id, record, client, model, use_parse, target_pairs):
    seen = set()
    rows = []

    def add(pairs):
        for pair in pairs:
            question = (pair.question or "").strip()
            answer = (pair.answer or "").strip()
            if question and answer and question not in seen:
                seen.add(question)
                rows.append({"question": question, "answer": answer, "id": pokemon_id})

    add(llm_structured_retry(client, model, record, use_parse))
    for _ in range(FILL_ATTEMPTS):
        if len(rows) >= target_pairs:
            break
        needed = target_pairs - len(rows)
        add(
            llm_structured_retry(
                client,
                model,
                record,
                use_parse,
                instructions=FILL_INSTRUCTIONS.replace("{needed}", str(needed)),
            )
        )
    return rows[:target_pairs]


def load_raw_records():
    if not RAW_POKEDEX.exists():
        raise SystemExit(
            f"FATAL: raw pokedex cache missing at {RAW_POKEDEX}. "
            "Run `uv run python -m src.data.ingest` first."
        )
    with open(RAW_POKEDEX, encoding="utf-8") as f:
        records = json.load(f)
    return {int(record["id"]): record for record in records}


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
        if len(selected) < size:  # defensive; cannot happen with 1,025 records
            selected.extend(remaining[: size - len(selected)])
    return selected


def coverage_summary(records):
    """Coverage stats for a selected subset: (types, generations, legendary, mythical)."""
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
        description="Generate LLM Pokémon Q&A pairs into evaluation/data/qa.jsonl "
        "(default: deterministic coverage-sampled dev subset — 50 records × 5 pairs)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="All 1,025 records × N pairs (MANUAL only — slow/costly)")
    group.add_argument("--limit", type=int, default=DEV_SUBSET_SIZE,
                       help=f"Coverage-sampled N records from the raw pokedex (default: {DEV_SUBSET_SIZE})")
    parser.add_argument("--pairs", type=int, default=TARGET_PAIRS_PER_RECORD,
                        help="Target Q&A pairs per record (default: 5; fewer = cheaper but weaker stats)")
    parser.add_argument("--seed", type=int, default=DEV_SUBSET_SEED,
                        help=f"Seed for the deterministic coverage sampler (default: {DEV_SUBSET_SEED})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip record ids already present in evaluation/data/qa.jsonl (safe re-run after a crash)")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")

    records = load_raw_records()
    ids = resolve_ids(records, args.limit, args.full, args.seed)

    if args.resume and QA_FILE.exists():
        existing = set()
        with open(QA_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["id"])
        before = len(ids)
        ids = [i for i in ids if i not in existing]
        print(f"Resume: skipped {before - len(ids)} ids already in {QA_FILE}")

    print(f"Records to generate for: {len(ids)}")
    if ids:
        sel_records = [records[i] for i in ids]
        types, gens, legendary, mythical = coverage_summary(sel_records)
        print(f"  coverage: {len(types)}/18 types, {len(gens)} generations, "
              f"{legendary} legendary, {mythical} mythical (seed={args.seed})")

    client = patch_openai_client(create_client())
    # Bound the SDK's default 600s per-request timeout and disable its built-in
    # retries (the llm_structured_retry loop below is the retry layer): a
    # black-holing endpoint must fail fast, not hang for minutes per attempt.
    client.timeout = 120.0
    client.max_retries = 0
    model = get_model()
    use_parse = supports_structured_output(client, model)
    mode = "parse" if use_parse else "create + json.loads (JSON fallback)"
    print(f"Model: {model} | structured mode: {mode} | expected pairs: {len(ids) * args.pairs}")

    all_rows = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [
                pool.submit(
                    generate_for_record, pokemon_id, records[pokemon_id], client, model, use_parse, args.pairs
                )
                for pokemon_id in ids
            ]
            with tqdm(total=len(futures), desc="Generating QA") as progress:
                for future in futures:
                    all_rows.extend(future.result())
                    progress.update()
    except Exception as exc:  # noqa: BLE001 — any worker/SDK failure → clean non-zero exit
        print(f"FATAL: QA generation failed: {exc}", file=sys.stderr)
        print(
            "No changes written to evaluation/data/qa.jsonl (written atomically only on success).",
            file=sys.stderr,
        )
        return 1

    all_rows.sort(key=lambda row: row["id"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_FILE, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows)
    print(f"Wrote {len(all_rows)} Q&A pairs to {QA_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

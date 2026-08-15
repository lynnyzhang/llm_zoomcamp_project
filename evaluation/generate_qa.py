# LLM-generated Pokémon ground-truth set (evaluation/data/qa.jsonl): per-record
# natural-language QUESTIONS linked to the Pokédex document that contains the
# answer. Mirrors the course's data-generation pattern for grounded QA sets:
# only questions are LLM-generated; the ground-truth answer is the document
# itself, looked up by id at eval time — the LLM never writes answers.
#
# Records come from the INDEXED documents (data/chunks/documents.jsonl) — the
# same dataset the retrievers index — so questions are grounded in exactly what
# retrieval sees (search_text, type_effectiveness, evolution links), not in the
# raw CSVs.

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from evaluation.dev_subset import (
    DEV_SUBSET_SIZE,
    DEV_SUBSET_SEED,
    coverage_summary,
    resolve_ids,
)
from evaluation.documents import load_pokemon_documents
from evaluation.evaluation_utils import map_progress
from evaluation.question_generator import (
    MAX_WORKERS,
    TARGET_QUESTIONS_PER_RECORD,
    generate_questions_for_record,
)
from src.llm import LLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_FILE = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate LLM ground-truth questions into evaluation/data/qa.jsonl "
        "(default: deterministic coverage-sampled dev subset — 50 Pokémon × 5 questions; "
        "each row links a question to the indexed Pokédex document that contains its answer)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="All 1,350 Pokémon documents × N questions (MANUAL only — slow/costly)")
    group.add_argument("--limit", type=int, default=DEV_SUBSET_SIZE,
                       help=f"Coverage-sampled N Pokémon documents from the indexed dataset (default: {DEV_SUBSET_SIZE})")
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

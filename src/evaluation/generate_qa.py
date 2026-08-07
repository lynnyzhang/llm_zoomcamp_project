"""LLM-generated Pokémon Q&A set (data/qa.jsonl).

Adapts the course 4-Evaluation data-generation pattern (reference only — the
technique is copied here; nothing is imported from 4-Evaluation/):

- per Pokémon record, prompt the LLM to formulate 5 natural fan questions
  plus answers grounded in the record (Responses API, structured output);
- pydantic structured output via ``client.responses.parse(text_format=...)``
  with a ``patch_openai_client``-style json_schema injection for local
  OpenAI-compatible servers (e.g. llama.cpp) and a plain
  ``responses.create`` + ``json.loads`` fallback for servers that reject
  ``parse`` entirely;
- 3 retries with 2^n backoff (``llm_structured_retry`` pattern);
- ThreadPoolExecutor (max_workers 2 — see MAX_WORKERS for the measured
  deviation from the plan's 6) + tqdm progress;
- writes ``data/qa.jsonl`` rows ``{"question": str, "answer": str, "id": int}``
  where id = Pokémon id (exact ground-truth linkage).

Default (dev subset): ids are read from ``data/corpus.jsonl`` — QA always
matches the indexed corpus (50 records × 5 = 250 pairs).
``--limit N``: first N records by id from the raw cache. ``--full``: all
1,025 records × 5 = 5,125 pairs — MANUAL only (user directive 2026-08-07:
never generate on the full set during execution).

data/qa.jsonl is written atomically: only after ALL records generated
successfully, so a failed run never leaves a partial file behind.
"""

from __future__ import annotations

import argparse
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
RAW_POKEDEX = DATA_DIR / "raw" / "complete_pokedex.json"
QA_FILE = DATA_DIR / "qa.jsonl"

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
    """Port of the course 4-Evaluation patch for local OpenAI-compatible servers.

    ``responses.parse`` sends a ``text`` field in the request body that
    llama.cpp-style servers reject; this injects the pydantic json_schema via
    ``extra_body.response_format`` instead and falls back to the native SDK
    parse when no ``text_format`` is given.
    """
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


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM completion (strips code fences)."""
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


def _repair_json(text: str) -> dict:
    """Light repairs for model JSON quirks, tried in order of likelihood.

    The local model frequently drops the outer closing brace
    (``{"qa_pairs": [...]``), sometimes adds trailing prose, and occasionally
    emits trailing commas — each variant below handles one combination.
    """
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


def _try_parse(client, model: str, messages: list[dict]) -> list[QAPair] | None:
    """parse(text_format=...) result, or None if the endpoint cannot produce it."""
    try:
        parsed = client.responses.parse(model=model, input=messages, text_format=QAResponse).output_parsed
    except Exception:  # noqa: BLE001 — servers that reject parse fall back to create
        return None
    return parsed.qa_pairs if parsed is not None else None


def _generate_qa_pairs(
    client, model: str, record: dict, use_parse: bool, instructions: str = DATA_GEN_INSTRUCTIONS
) -> list[QAPair]:
    """One structured generation attempt: parse() with a create() fallback."""
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


def supports_structured_output(client, model: str) -> bool:
    """Probe once whether the endpoint honors parse(text_format=...).

    Local OpenAI-compatible servers (e.g. llama.cpp) commonly accept the
    request but return plain text (output_parsed=None), silently wasting a
    full generation per record — so we probe cheaply and skip parse entirely
    when it is not honored.
    """
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
    model: str,
    record: dict,
    use_parse: bool,
    max_retries: int = 3,
    instructions: str = DATA_GEN_INSTRUCTIONS,
) -> list[QAPair]:
    """Try structured generation up to ``max_retries`` times with 2^n backoff."""
    for attempt in range(max_retries):
        try:
            return _generate_qa_pairs(client, model, record, use_parse, instructions)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


def generate_for_record(
    pokemon_id: int, record: dict, client, model: str, use_parse: bool
) -> list[dict]:
    """Generate the Q&A rows for one Pokémon record (id = ground truth).

    Rows are deduplicated by question, topped up to TARGET_PAIRS_PER_RECORD
    with follow-up generations when the model returns fewer, and capped at the
    target so every record contributes exactly 5 rows.
    """
    seen: set[str] = set()
    rows: list[dict] = []

    def add(pairs: list[QAPair]) -> None:
        for pair in pairs:
            question = (pair.question or "").strip()
            answer = (pair.answer or "").strip()
            if question and answer and question not in seen:
                seen.add(question)
                rows.append({"question": question, "answer": answer, "id": pokemon_id})

    add(llm_structured_retry(client, model, record, use_parse))
    for _ in range(FILL_ATTEMPTS):
        if len(rows) >= TARGET_PAIRS_PER_RECORD:
            break
        needed = TARGET_PAIRS_PER_RECORD - len(rows)
        add(
            llm_structured_retry(
                client,
                model,
                record,
                use_parse,
                instructions=FILL_INSTRUCTIONS.replace("{needed}", str(needed)),
            )
        )
    return rows[:TARGET_PAIRS_PER_RECORD]


def load_raw_records() -> dict[int, dict]:
    """Load the full pokedex cache (todo 1's deterministic source), keyed by id."""
    if not RAW_POKEDEX.exists():
        raise SystemExit(
            f"FATAL: raw pokedex cache missing at {RAW_POKEDEX}. "
            "Run `uv run python -m src.data.ingest` first."
        )
    with open(RAW_POKEDEX, encoding="utf-8") as f:
        records = json.load(f)
    return {int(record["id"]): record for record in records}


def load_corpus_ids() -> list[int]:
    """The exact id set indexed in corpus.jsonl (QA must match the corpus)."""
    if not CORPUS_FILE.exists():
        raise SystemExit(
            f"FATAL: corpus missing at {CORPUS_FILE}. "
            "Run `uv run python -m src.data.ingest` first."
        )
    ids = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    if not ids:
        raise SystemExit(
            f"FATAL: {CORPUS_FILE} contains no records — "
            "run `uv run python -m src.data.ingest` first."
        )
    return sorted(ids)


def resolve_ids(
    records: dict[int, dict], limit: int | None, full: bool, corpus_ids: list[int]
) -> list[int]:
    """Dev subset (default) = corpus ids; --limit N / --full = slice of the raw cache."""
    if full:
        return sorted(records)
    if limit is not None:
        return sorted(records)[:limit]
    return corpus_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate LLM Pokémon Q&A pairs into data/qa.jsonl "
        "(default: the exact id set in data/corpus.jsonl = 50 records × 5 pairs)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full",
        action="store_true",
        help="All 1,025 records × 5 = 5,125 pairs (MANUAL only — slow/costly)",
    )
    group.add_argument(
        "--limit",
        type=int,
        help="First N records by id from the raw cache (default: corpus.jsonl ids)",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")

    records = load_raw_records()
    corpus_ids = load_corpus_ids() if not (args.full or args.limit is not None) else []
    ids = resolve_ids(records, args.limit, args.full, corpus_ids)
    print(f"Records to generate for: {len(ids)} (ids {ids[0]}..{ids[-1]})")

    client = patch_openai_client(create_client())
    # Bound the SDK's default 600s per-request timeout and disable its built-in
    # retries (the llm_structured_retry loop below is the retry layer): a
    # black-holing endpoint must fail fast, not hang for minutes per attempt.
    client.timeout = 120.0
    client.max_retries = 0
    model = get_model()
    use_parse = supports_structured_output(client, model)
    mode = "parse" if use_parse else "create + json.loads (JSON fallback)"
    print(f"Model: {model} | structured mode: {mode} | expected pairs: {len(ids) * 5}")

    all_rows: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [
                pool.submit(
                    generate_for_record, pokemon_id, records[pokemon_id], client, model, use_parse
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
            "No changes written to data/qa.jsonl (written atomically only on success).",
            file=sys.stderr,
        )
        return 1

    all_rows.sort(key=lambda row: row["id"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(QA_FILE, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows)
    print(f"Wrote {len(all_rows)} Q&A pairs to {QA_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

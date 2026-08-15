import json
import time

from pydantic import BaseModel

from evaluation.llm_calls import llm_structured

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

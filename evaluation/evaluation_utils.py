import json
import time
import types
from pathlib import Path

from tqdm.auto import tqdm

from src.llm import LLMClient

DEFAULT_DOCUMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "chunks" / "documents.jsonl"
)


def calc_price(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model=None):
    model = model or LLMClient.get_model()
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )
    return response.output_parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model=None,
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results


def load_document_index(documents_path=None):
    """data/chunks/documents.jsonl -> {str(doc id): doc}.

    The ground-truth answer for a QA row is the linked document's content —
    the analog of the FAQ 'answer' lookup in the course's RAG eval
    (answer_orig = doc_idx[doc_id]["answer"]).
    """
    path = Path(documents_path) if documents_path else DEFAULT_DOCUMENTS_PATH
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                docs[str(doc["id"])] = doc
    return docs


def ground_truth_answer(doc_idx, document_id):
    """The linked document's content — the ground-truth answer for a QA row.

    Pokémon documents carry the answer inside search_text (the labeled record
    rendering), the analog of the FAQ 'answer' field in the course's evals.
    """
    doc = doc_idx.get(str(document_id))
    return doc.get("search_text") if doc else None

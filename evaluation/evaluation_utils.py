# evaluation/evaluation_utils.py
#
# Shared evaluation helpers, mirroring the course's evaluation_utils.py as
# closely as possible (calc_price, calc_total_price, patch_openai_client,
# llm_structured, llm_structured_retry, map_progress) plus the
# Pokémon-specific document-index lookup used by the LLM-judge evals.

import json
import time
import types
from pathlib import Path
from pydantic import BaseModel
from tqdm.auto import tqdm
from src.llm import get_model

_DEFAULT_DOCUMENTS_PATH = (
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


def patch_openai_client(client):
    """
    Safely overrides client.responses.parse to support llama.cpp via
    'response_format' while falling back cleanly to the native SDK function
    when no text_format is requested.
    """
    _original_parse = client.responses.parse

    def _patched_responses_parse(self, model, input, **kwargs):
        pydantic_model = kwargs.pop("text_format", None)

        if pydantic_model is None:
            return _original_parse(model=model, input=input, **kwargs)

        llama_cpp_format = {
            "type": "json_schema",
            "json_schema": {
                "name": pydantic_model.__name__,
                "strict": True,
                "schema": pydantic_model.model_json_schema(),
            },
        }

        # Inject the formatting schema into the extra_body container payload
        extra_body = kwargs.pop("extra_body", {})
        extra_body["response_format"] = llama_cpp_format

        return _original_parse(
            model=model,
            input=input,
            extra_body=extra_body,
            **kwargs,
        )

    client.responses.parse = types.MethodType(_patched_responses_parse, client.responses)
    return client


def supports_text_format_parse(client, model, output_type):
    # Local servers (e.g. llama.cpp) accept the request but return plain text
    # (output_parsed=None), silently wasting a generation — probe once and patch if none.
    try:
        probe = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": 'Reply with JSON only: {"qa_pairs": []}'}],
            text_format=output_type
        )
        return probe.output_parsed is not None
    except Exception:  # noqa: BLE001 — any rejection means JSON fallback mode
        return False


def llm_structured(client, instructions, user_prompt, output_type, model=None):
    model = model or get_model()
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    if not supports_text_format_parse(client, model, output_type):
        client = patch_openai_client(client)

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
        except Exception:  # noqa: BLE001 — retry covers any transient failure
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
    path = Path(documents_path) if documents_path else _DEFAULT_DOCUMENTS_PATH
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

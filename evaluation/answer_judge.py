from pydantic import BaseModel

from evaluation.document_index import ground_truth_answer
from evaluation.llm_calls import llm_structured
from src.llm_client import LLMClient


class JudgeScore(BaseModel):
    score: int
    explanation: str


def llm_judge_score(client, question, generated_answer, ground_truth, model=None):
    model = model or LLMClient.get_model()
    instructions = """\
You are an expert judge for question-answering systems.
Given a question, generated answer, and ground truth answer, rate the
generated answer's correctness on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Mostly wrong with some relevant info
3 = Partially correct
4 = Mostly correct with minor issues
5 = Fully correct and complete

Respond ONLY with a JSON object: {"score": <int>, "explanation": "<brief explanation>"}
"""

    prompt = f"""\
Question: {question}

Generated Answer: {generated_answer}

Ground Truth Answer: {ground_truth}
"""

    # Single path: structured output via responses.parse(text_format=JudgeScore).
    # The eval scripts use their own patch_openai_client (evaluation_utils):
    # works on OpenAI (native structured output) AND on llama.cpp
    # (response_format via extra_body). There is deliberately NO create()/JSON
    # extraction fallback: if the backend cannot honor the schema, the output
    # is prose — not JSON — so there is nothing to salvage; fail loudly.
    try:
        parsed, _ = llm_structured(client, instructions, prompt, JudgeScore, model=model)
        if parsed is None:
            raise ValueError(
                "structured output unsupported: server returned output_parsed=None"
            )
        return {"score": parsed.score, "explanation": parsed.explanation}
    except Exception as e:
        print(f"  Judge error: {e}")
        return None


def evaluate_answer_quality(
    rag_fn,
    client,
    questions,
    sample_size=50,
    model=None,
    doc_idx=None,
):
    model = model or LLMClient.get_model()
    pairs = questions[:sample_size] if sample_size > 0 else questions
    scores = []
    errors = 0

    for i, pair in enumerate(pairs):
        question = pair["question"]
        # Ground truth is the linked document's own content (course RAG-eval
        # pattern: answer_orig = doc_idx[doc_id]["answer"]) — never an
        # LLM-written answer from generation time.
        ground_truth = ground_truth_answer(doc_idx, pair["document"]) if doc_idx else None
        if ground_truth is None:
            print(f"  [{i+1}/{len(pairs)}] No ground-truth document for question {pair.get('document')}")
            errors += 1
            continue

        try:
            generated = rag_fn(question)
        except Exception as e:  # noqa: BLE001 — per-question errors never abort the batch
            print(f"  [{i+1}/{len(pairs)}] RAG error: {e}")
            errors += 1
            continue

        result = llm_judge_score(client, question, generated, ground_truth, model=model)
        if result is None:
            errors += 1
            continue

        result["question_id"] = pair.get("id")
        result["question"] = question
        scores.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] evaluated")

    if not scores:
        return {"mean_score": 0, "scores": [], "num_evaluated": 0, "errors": errors}

    n = len(scores)
    return {
        "mean_score": round(sum(s["score"] for s in scores) / n, 2),
        "scores": scores,
        "num_evaluated": n,
        "errors": errors,
    }

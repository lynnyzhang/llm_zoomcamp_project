import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_utils import (
    ground_truth_answer,
    load_document_index,
)
from src.llm import LLMClient

# ---------------------------------------------------------------------------
# Pydantic model for structured judge output
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    faithfulness: int  # 1-5
    relevance: int     # 1-5
    coherence: int     # 1-5
    explanation: str   # brief reasoning


# ---------------------------------------------------------------------------
# Judge prompts (3 variants to compare)
# ---------------------------------------------------------------------------

JUDGE_PROMPTS = {
    "simple": {
        "instructions": "You are an AI judge. Rate the answer on a 1-5 scale.",
        "template": """\
Question: {question}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}

Rate faithfulness (context support), relevance (addresses question), coherence (clarity). 1-5 each.
Respond with a single JSON object in exactly this format (no markdown, no extra text):
{{"faithfulness": 1-5, "relevance": 1-5, "coherence": 1-5, "explanation": "brief reasoning"}}""",
    },
    "detailed": {
        "instructions": """\
You are an expert evaluator for question-answering systems.
Rate the generated answer on three dimensions using a 1-5 scale:
- Faithfulness (1-5): How well is the answer supported by the provided context? 5 = fully grounded, 1 = contradicts context.
- Relevance (1-5): Does the answer actually address the question asked? 5 = directly answers, 1 = off-topic.
- Coherence (1-5): Is the answer well-structured and clear? 5 = excellent, 1 = incoherent.
Provide a brief explanation for your ratings.""",
        "template": """\
Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Evaluate the generated answer against the context and ground truth.
Provide scores (1-5) for faithfulness, relevance, and coherence.
Respond with a single JSON object in exactly this format (no markdown, no extra text):
{{"faithfulness": 1-5, "relevance": 1-5, "coherence": 1-5, "explanation": "brief reasoning"}}""",
    },
    "with_examples": {
        "instructions": """\
You are an expert evaluator for question-answering systems.

Examples of ratings:

Example 1:
Q: "What type is Pikachu?"
Context: "Pikachu is an electric-type Pokémon."
Answer: "Pikachu is an electric type."
Ground Truth: "Pikachu is an electric type."
Rating: faithfulness=5, relevance=5, coherence=5

Example 2:
Q: "What type is Pikachu?"
Context: "Pikachu is an electric-type Pokémon."
Answer: "Pikachu is a grass type."
Ground Truth: "Pikachu is an electric type."
Rating: faithfulness=1, relevance=2, coherence=4

Now evaluate the following:""",
        "template": """\
Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Rate faithfulness (1-5), relevance (1-5), coherence (1-5). Explain briefly.
Respond with a single JSON object in exactly this format (no markdown, no extra text):
{{"faithfulness": 1-5, "relevance": 1-5, "coherence": 1-5, "explanation": "brief reasoning"}}""",
    },
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def llm_judge(client, instructions, user_prompt, model=None):
    model = model or LLMClient.get_model()
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    # Single path: structured output via responses.parse(text_format=JudgeScore).
    # The eval scripts use their own patch_openai_client (evaluation_utils)
    # for llama.cpp (response_format via extra_body), passing through to the
    # native SDK on OpenAI. There is deliberately NO
    # create()/JSON-extraction fallback: if the backend cannot honor the
    # schema, the output is prose — not JSON — so there is nothing to salvage
    # (llm_judge_retry retries transient failures).
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=JudgeScore,
    )
    if response.output_parsed is None:
        raise ValueError("structured output unsupported: server returned output_parsed=None")
    return response.output_parsed


def llm_judge_retry(client, instructions, user_prompt, model=None, max_retries=3):
    model = model or LLMClient.get_model()
    for attempt in range(max_retries):
        try:
            return llm_judge(client, instructions, user_prompt, model=model)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Judge call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_qa_pairs(qa_path):
    pairs = []
    with open(qa_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def evaluate_single(
    client,
    question,
    context,
    generated_answer,
    ground_truth,
    judge_config,
    model=None,
):
    model = model or LLMClient.get_model()
    user_prompt = judge_config["template"].format(
        question=question,
        context=context,
        answer=generated_answer,
        ground_truth=ground_truth,
    )
    result = llm_judge_retry(client, judge_config["instructions"], user_prompt, model=model)
    if result is None:
        return None
    return {
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "coherence": result.coherence,
        "explanation": result.explanation,
    }


def evaluate_with_prompt(
    rag_pipeline,
    client,
    qa_pairs,
    judge_config,
    model=None,
    sample_size=50,
    doc_idx=None,
):
    model = model or LLMClient.get_model()
    pairs = qa_pairs[:sample_size] if sample_size > 0 else qa_pairs
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
            search_results = rag_pipeline.search(question)
            context = rag_pipeline.build_context(search_results)
            generated = rag_pipeline.rag(question)
        except Exception as e:
            print(f"  [{i+1}/{len(pairs)}] RAG error for question {pair.get('id')}: {e}")
            errors += 1
            continue

        result = evaluate_single(
            client, question, context, generated, ground_truth, judge_config, model=model,
        )
        if result is None:
            errors += 1
            continue

        result["question_id"] = pair.get("id")
        result["question"] = question
        result["generated_answer"] = generated[:200]  # truncate for storage
        scores.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] processed")

    if not scores:
        return {"mean": {"faithfulness": 0, "relevance": 0, "coherence": 0}, "samples": [], "errors": errors}

    n = len(scores)
    mean = {
        "faithfulness": round(sum(s["faithfulness"] for s in scores) / n, 2),
        "relevance": round(sum(s["relevance"] for s in scores) / n, 2),
        "coherence": round(sum(s["coherence"] for s in scores) / n, 2),
    }

    return {
        "mean": mean,
        "samples": scores,
        "num_evaluated": n,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv

    from src.llm import LLMClient

    load_dotenv(PROJECT_ROOT / ".env")

    qa_path = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "llm_eval.json"

    print("Loading Q&A pairs...")
    qa_pairs = load_qa_pairs(str(qa_path))
    print(f"Loaded {len(qa_pairs)} pairs")

    print("Loading document index (ground-truth answers)...")
    doc_idx = load_document_index()
    print(f"Loaded {len(doc_idx)} documents")

    client = LLMClient.get()

    print("Initializing RAG pipeline...")
    t0 = time.time()
    from src.rag.RAGBase import RAGBase
    from src.search.hybrid import HybridSearch

    search_index = HybridSearch()
    rag = RAGBase(search_index=search_index, llm_client=client)
    print(f"RAG pipeline ready in {time.time() - t0:.1f}s")

    # Sample for evaluation (use 50 for speed, set to 0 for full)
    sample_size = 10

    all_results = {}

    for name, config in JUDGE_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating with '{name}' judge prompt...")
        print(f"{'='*60}")

        t0 = time.time()
        result = evaluate_with_prompt(
            rag, client, qa_pairs, config,
            sample_size=sample_size,
            doc_idx=doc_idx,
        )
        elapsed = time.time() - t0
        result["time_seconds"] = round(elapsed, 2)
        result["prompt_name"] = name
        all_results[name] = result

        print(f"  Mean faithfulness: {result['mean']['faithfulness']}")
        print(f"  Mean relevance:    {result['mean']['relevance']}")
        print(f"  Mean coherence:    {result['mean']['coherence']}")
        print(f"  Evaluated: {result.get('num_evaluated', 0)}, Errors: {result['errors']}")
        print(f"  Time: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("PROMPT COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Prompt':<18} {'Faith':>6} {'Rel':>6} {'Coh':>6} {'Time':>8}")
    print("-" * 60)
    for name, result in all_results.items():
        m = result["mean"]
        print(
            f"{name:<18} {m['faithfulness']:>6.2f} {m['relevance']:>6.2f} "
            f"{m['coherence']:>6.2f} {result['time_seconds']:>7.1f}s"
        )
    print("=" * 60)

    best_name = max(
        all_results.keys(),
        key=lambda k: (
            all_results[k]["mean"]["faithfulness"]
            + all_results[k]["mean"]["relevance"]
            + all_results[k]["mean"]["coherence"]
        ) / 3,
    )
    print(f"\nBest prompt: {best_name}")

    # Save results (strip per-sample details to keep file manageable)
    save_data = {}
    for name, result in all_results.items():
        save_data[name] = {
            "faithfulness": result["mean"]["faithfulness"],
            "relevance": result["mean"]["relevance"],
            "coherence": result["mean"]["coherence"],
            "num_evaluated": result.get("num_evaluated", 0),
            "errors": result["errors"],
            "time_seconds": result["time_seconds"],
            "prompt_name": name,
            "sample_examples": result.get("samples", [])[:5],
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

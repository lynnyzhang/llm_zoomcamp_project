import sys
from pathlib import Path

from pydantic import BaseModel

from src.llm import LLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class JudgeScore(BaseModel):
    faithfulness: int  # 1-5
    relevance: int     # 1-5
    coherence: int     # 1-5
    explanation: str   # brief reasoning


def evaluate_with_prompt(
    rag_pipeline,
    client,
    qa_pairs,
    judge_config,
    model=None,
    sample_size=50,
    doc_idx=None,
):
    from evaluation.documents import ground_truth_answer
    from evaluation.llm_judge import evaluate_single

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


def main():
    from evaluation.llm_eval_flow import (
        evaluate_all_prompts,
        print_summary,
        save_results,
        setup_rag,
    )

    qa_pairs, doc_idx, client, rag = setup_rag()

    sample_size = 10

    all_results = evaluate_all_prompts(rag, client, qa_pairs, doc_idx, sample_size)
    print_summary(all_results)

    output_path = PROJECT_ROOT / "evaluation" / "results" / "llm_eval.json"
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()

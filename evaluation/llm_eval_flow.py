import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.llm import LLMClient
from src.rag.RAGBase import RAGBase
from src.search.hybrid import HybridSearch


def setup_rag():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / ".env")

    from evaluation.documents import load_document_index
    from evaluation.retrieval_metrics import load_qa_pairs

    qa_path = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading Q&A pairs...")
    qa_pairs = load_qa_pairs(str(qa_path))
    print(f"Loaded {len(qa_pairs)} pairs")

    print("Loading document index (ground-truth answers)...")
    doc_idx = load_document_index()
    print(f"Loaded {len(doc_idx)} documents")

    client = LLMClient.get()

    print("Initializing RAG pipeline...")
    t0 = time.time()
    search_index = HybridSearch()
    rag = RAGBase(search_index=search_index, llm_client=client)
    print(f"RAG pipeline ready in {time.time() - t0:.1f}s")

    return qa_pairs, doc_idx, client, rag


def evaluate_all_prompts(rag, client, qa_pairs, doc_idx, sample_size):
    from evaluation.judge_prompts import JUDGE_PROMPTS
    from evaluation.llm_eval import evaluate_with_prompt

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

    return all_results


def print_summary(all_results):
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


def save_results(all_results, output_path):
    # Strip per-sample details to keep the output file manageable.
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

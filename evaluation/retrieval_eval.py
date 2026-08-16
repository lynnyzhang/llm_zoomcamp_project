import json
import sys
import time
from pathlib import Path

from evaluation.retrieval_metrics import load_qa_pairs, mrr, precision_at_k, recall_at_k

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_search(search_fn, questions, k=5):
    precisions = []
    recalls = []
    mrrs = []

    for item in questions:
        query = item["question"]
        relevant_id = str(item["document"])

        results = search_fn(query, num_results=k)
        retrieved_ids = [str(doc.id) for doc in results]

        precisions.append(precision_at_k(retrieved_ids, relevant_id, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr(retrieved_ids, relevant_id))

    n = len(questions)
    return {
        f"precision@{k}": round(sum(precisions) / n, 4),
        f"recall@{k}": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "num_questions": n,
    }


def main():
    from src.search.hybrid_search import HybridSearch

    qa_path = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "retrieval_eval.json"

    print("Loading ground truth...")
    questions = load_qa_pairs(str(qa_path))
    print(f"Loaded {len(questions)} questions")

    print("Initializing search indices...")
    t0 = time.time()
    hs = HybridSearch()
    print(f"Search indices built in {time.time() - t0:.1f}s")

    search_methods = {
        "keyword": hs.keyword_search,
        "vector": hs.vector_search,
        "hybrid": hs.search,
    }

    k = 5
    all_results = {}

    for name, search_fn in search_methods.items():
        print(f"\nEvaluating {name} search...")
        t0 = time.time()
        scores = evaluate_search(search_fn, questions, k=k)
        elapsed = time.time() - t0
        scores["time_seconds"] = round(elapsed, 2)
        all_results[name] = scores
        print(f"  precision@{k}: {scores[f'precision@{k}']}")
        print(f"  recall@{k}: {scores[f'recall@{k}']}")
        print(f"  mrr: {scores['mrr']}")
        print(f"  time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"{'Method':<12} {'P@5':>8} {'R@5':>8} {'MRR':>8} {'Time':>8}")
    print("-" * 60)
    for name, scores in all_results.items():
        print(
            f"{name:<12} {scores[f'precision@{k}']:>8.4f} "
            f"{scores[f'recall@{k}']:>8.4f} {scores['mrr']:>8.4f} "
            f"{scores['time_seconds']:>7.1f}s"
        )
    print("=" * 60)

    # Report the best method factually — no claim that hybrid must win: on
    # the Pokémon dev subset, exact-name keyword search often beats vector.
    best_method = max(all_results, key=lambda m: all_results[m][f"precision@{k}"])
    print(f"\nBest method by precision@{k}: {best_method}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

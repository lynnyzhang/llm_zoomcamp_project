import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_ground_truth(qa_path):
    questions = []
    with open(qa_path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def precision_at_k(retrieved_ids, relevant_id, k):
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id == relevant_id)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids, relevant_id, k):
    top_k = retrieved_ids[:k]
    return 1.0 if relevant_id in top_k else 0.0


def mrr(retrieved_ids, relevant_id):
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id == relevant_id:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_search(search_fn, questions, k=5):
    precisions = []
    recalls = []
    mrrs = []

    for item in questions:
        query = item["question"]
        relevant_id = str(item["id"])

        results = search_fn(query, num_results=k)
        retrieved_ids = [str(doc.get("id", "")) for doc in results]

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
    from src.search.hybrid import HybridSearch

    # Paths
    qa_path = PROJECT_ROOT / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "retrieval_eval.json"

    # Load ground truth
    print("Loading ground truth...")
    questions = load_ground_truth(str(qa_path))
    print(f"Loaded {len(questions)} questions")

    # Initialize hybrid search (loads documents + builds indices)
    print("Initializing search indices...")
    t0 = time.time()
    hs = HybridSearch()
    print(f"Search indices built in {time.time() - t0:.1f}s")

    # Define search functions
    search_methods = {
        "keyword": hs.keyword_search,
        "vector": hs.vector_search,
        "hybrid": hs.search,
    }

    # Evaluate each method
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

    # Summary
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

    # Determine best
    best_hybrid = all_results["hybrid"][f"precision@{k}"] > all_results["keyword"][f"precision@{k}"]
    print(f"\nHybrid > Keyword precision@{k}: {best_hybrid}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

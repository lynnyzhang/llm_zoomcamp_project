import json


def load_qa_pairs(qa_path):
    pairs = []
    with open(qa_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


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


def retrieval_accuracy(search_fn, questions, k=5):
    hits = 0
    details = []

    for item in questions:
        query = item["question"]
        relevant_id = str(item["document"])

        results = search_fn(query, num_results=k)
        retrieved_ids = [str(doc.get("id", "")) for doc in results]
        hit = relevant_id in retrieved_ids
        hits += hit
        details.append({
            "question_id": item["document"],
            "hit": hit,
            "retrieved_ids": retrieved_ids[:k],
        })

    n = len(questions)
    return {
        "hit_rate": round(hits / n, 4),
        "hits": hits,
        "total": n,
        "details": details,
    }

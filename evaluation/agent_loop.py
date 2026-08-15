import time

import numpy as np


def run_agent_loop(agent, rag_base, questions, llm_available, show_avg=False):
    search_counts = []
    hits = 0
    latencies = []

    from src.rag.tools import SearchRecord

    for i, pair in enumerate(questions):
        query = pair["question"]
        relevant_id = str(pair["document"])

        q_start = time.time()
        if llm_available:
            result = agent.run(query)
            search_counts.append(result["iterations"])
        else:
            search_results = rag_base.search(query)
            search_counts.append(1)
            result = {
                "searches": [SearchRecord(query, search_results, source="local")],
                "answer": "",
                "iterations": 1,
                "rejected": False,
                "source": "local",
                "confidence": None,
            }
        latencies.append(time.time() - q_start)

        # Only LOCAL search records carry doc ids — web results (title/url/
        # snippet) have none, so they cannot count toward doc-id retrieval.
        all_ids = []
        for search_record in result["searches"]:
            if search_record.source != "local":
                continue
            for doc in search_record.results:
                all_ids.append(str(doc.get("id", "")))

        if relevant_id in all_ids:
            hits += 1

        if (i + 1) % 10 == 0:
            if show_avg:
                print(f"  [{i+1}/{len(questions)}] processed, avg searches: {np.mean(search_counts):.2f}")
            else:
                print(f"  [{i+1}/{len(questions)}] processed")

    return search_counts, hits, latencies


def judge_agent_answers(agent, client, questions, doc_idx, llm_available, judge_sample):
    from evaluation.answer_judge import llm_judge_score
    from evaluation.documents import ground_truth_answer

    if not llm_available:
        return [], 0, judge_sample

    print("\nEvaluating agent answer quality (LLM judge)...")
    agent_quality_scores = []
    agent_errors = 0

    for i, pair in enumerate(questions):
        question = pair["question"]
        ground_truth = ground_truth_answer(doc_idx, pair["document"])
        if ground_truth is None:
            print(f"  [{i+1}/{len(questions)}] No ground-truth document for question {pair.get('document')}")
            agent_errors += 1
            continue

        try:
            result = agent.run(question)
            generated = result["answer"]
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] Error: {e}")
            agent_errors += 1
            continue

        judge_result = llm_judge_score(client, question, generated, ground_truth)
        if judge_result is None:
            agent_errors += 1
            continue

        judge_result["question_id"] = pair.get("id")
        judge_result["question"] = question
        judge_result["source"] = result.get("source")
        judge_result["confidence"] = result.get("confidence")
        agent_quality_scores.append(judge_result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] evaluated")

    agent_mean_score = round(np.mean([s["score"] for s in agent_quality_scores]), 2) if agent_quality_scores else 0

    print(f"  Mean score: {agent_mean_score}/5")
    print(f"  Evaluated: {len(agent_quality_scores)}, Errors: {agent_errors}")

    return agent_quality_scores, agent_mean_score, agent_errors

import time

import numpy as np

from evaluation.answer_judge import evaluate_answer_quality
from evaluation.retrieval_metrics import retrieval_accuracy


def run_simple_rag(rag_base, client, qa_pairs, doc_idx, judge_sample, llm_available):
    print("\n" + "=" * 60)
    print("PHASE 1: Simple RAG (single search)")
    print("=" * 60)

    print("Measuring retrieval accuracy...")
    t0 = time.time()
    simple_retrieval = retrieval_accuracy(rag_base.search, qa_pairs, k=5)
    simple_retrieval_time = time.time() - t0
    print(f"  Hit rate: {simple_retrieval['hit_rate']:.1%} ({simple_retrieval['hits']}/{simple_retrieval['total']})")
    print(f"  Time: {simple_retrieval_time:.1f}s")

    simple_latency_per_query = 0.0

    if llm_available:
        print("\nEvaluating answer quality (LLM judge)...")
        t0 = time.time()
        simple_quality = evaluate_answer_quality(
            rag_base.rag, client, qa_pairs, sample_size=judge_sample, doc_idx=doc_idx,
        )
        simple_latency = time.time() - t0
        print(f"  Mean score: {simple_quality['mean_score']}/5")
        print(f"  Evaluated: {simple_quality['num_evaluated']}, Errors: {simple_quality['errors']}")
        print(f"  Total latency: {simple_latency:.1f}s")
        simple_latency_per_query = simple_latency / max(simple_quality["num_evaluated"], 1)
    else:
        simple_quality = {"mean_score": 0, "num_evaluated": 0, "errors": judge_sample}

    return {
        "simple_retrieval": simple_retrieval,
        "simple_retrieval_time": simple_retrieval_time,
        "simple_quality": simple_quality,
        "simple_latency_per_query": simple_latency_per_query,
    }


def run_agentic_rag(agent, rag_base, client, qa_pairs, doc_idx, judge_sample, llm_available):
    from evaluation.agent_loop import judge_agent_answers, run_agent_loop

    print("\n" + "=" * 60)
    print("PHASE 2: Agentic RAG (agent loop)")
    print("=" * 60)

    print("Running agent loop on sample...")
    agent_sample = qa_pairs[:judge_sample]
    agent_search_counts, agent_retrieval_hits, agent_latencies = run_agent_loop(
        agent, rag_base, agent_sample, llm_available, show_avg=True,
    )

    avg_searches = np.mean(agent_search_counts)
    avg_latency = np.mean(agent_latencies)
    agent_retrieval_rate = agent_retrieval_hits / len(agent_sample)

    print(f"\n  Agent stats (n={len(agent_sample)}):")
    print(f"    Avg searches/query: {avg_searches:.2f}")
    print(f"    Avg latency/query: {avg_latency:.2f}s")
    print(f"    Retrieval hit rate: {agent_retrieval_rate:.1%}")

    agent_quality_scores, agent_mean_score, agent_errors = judge_agent_answers(
        agent, client, agent_sample, doc_idx, llm_available, judge_sample,
    )

    return {
        "agent_sample": agent_sample,
        "avg_searches": avg_searches,
        "avg_latency": avg_latency,
        "agent_retrieval_rate": agent_retrieval_rate,
        "agent_quality_scores": agent_quality_scores,
        "agent_mean_score": agent_mean_score,
        "agent_errors": agent_errors,
    }


def run_retrieval_comparison(agent, rag_base, qa_pairs, agent_full_sample, llm_available):
    from evaluation.agent_loop import run_agent_loop

    print("\n" + "=" * 60)
    print("PHASE 3: Full retrieval comparison (dev subset)")
    print("=" * 60)

    full_sample = qa_pairs[:agent_full_sample]
    print(f"Running agent on {len(full_sample)} questions for retrieval metrics...")
    t0 = time.time()
    full_agent_search_counts, full_agent_hits, _ = run_agent_loop(
        agent, rag_base, full_sample, llm_available,
    )
    full_agent_time = time.time() - t0
    full_agent_retrieval_rate = full_agent_hits / len(full_sample)
    full_avg_searches = np.mean(full_agent_search_counts)

    print(f"  Agent full retrieval hit rate: {full_agent_retrieval_rate:.1%}")
    print(f"  Avg searches/query: {full_avg_searches:.2f}")
    print(f"  Total time: {full_agent_time:.1f}s")

    return {
        "full_sample": full_sample,
        "full_agent_time": full_agent_time,
        "full_agent_retrieval_rate": full_agent_retrieval_rate,
        "full_agent_hits": full_agent_hits,
        "full_avg_searches": full_avg_searches,
    }

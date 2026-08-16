import json

from src.llm_client import LLMClient


def build_results(qa_pairs, judge_sample, simple, agentic, comp):
    results = {
        "simple_rag": {
            "accuracy": simple["simple_retrieval"]["hit_rate"],
            "retrieval": {
                "hit_rate": simple["simple_retrieval"]["hit_rate"],
                "hits": simple["simple_retrieval"]["hits"],
                "total": simple["simple_retrieval"]["total"],
            },
            "answer_quality": {
                "mean_score": simple["simple_quality"]["mean_score"],
                "num_evaluated": simple["simple_quality"]["num_evaluated"],
            },
            "latency_per_query": round(simple["simple_latency_per_query"], 2),
            "retrieval_time_seconds": round(simple["simple_retrieval_time"], 2),
        },
        "agentic_rag": {
            "accuracy": round(comp["full_agent_retrieval_rate"], 4),
            "retrieval": {
                "hit_rate": round(comp["full_agent_retrieval_rate"], 4),
                "hits": comp["full_agent_hits"],
                "total": len(comp["full_sample"]),
            },
            "answer_quality": {
                "mean_score": agentic["agent_mean_score"],
                "num_evaluated": len(agentic["agent_quality_scores"]),
            },
            "avg_searches_per_query": round(float(comp["full_avg_searches"]), 2),
            "latency_per_query": round(float(agentic["avg_latency"]), 2),
            "total_time_seconds": round(comp["full_agent_time"], 2),
        },
        "comparison": {
            "retrieval_improvement": round(comp["full_agent_retrieval_rate"] - simple["simple_retrieval"]["hit_rate"], 4),
            "answer_quality_improvement": round(agentic["agent_mean_score"] - simple["simple_quality"]["mean_score"], 2),
            "latency_overhead": round(float(agentic["avg_latency"] - simple["simple_latency_per_query"]), 2),
            "search_overhead": round(float(comp["full_avg_searches"] - 1.0), 2),
        },
        "config": {
            "total_questions": len(qa_pairs),
            "judge_sample_size": judge_sample,
            "model": LLMClient.get_model(),
        },
    }
    return results


def print_summary(results):
    r = results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Simple RAG retrieval hit rate:  {r['simple_rag']['retrieval']['hit_rate']:.1%}")
    print(f"  Agent RAG retrieval hit rate:   {r['agentic_rag']['retrieval']['hit_rate']:.1%}")
    print(f"  Retrieval improvement:          {r['comparison']['retrieval_improvement']:+.1%}")
    print(f"  Simple RAG answer score:        {r['simple_rag']['answer_quality']['mean_score']}/5")
    print(f"  Agent RAG answer score:         {r['agentic_rag']['answer_quality']['mean_score']}/5")
    print(f"  Answer quality improvement:     {r['comparison']['answer_quality_improvement']:+.2f}")
    print(f"  Avg searches/query (agent):     {r['agentic_rag']['avg_searches_per_query']}")
    print(f"  Latency overhead:               {r['comparison']['latency_overhead']:+.2f}s/query")


def write_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

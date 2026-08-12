import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_utils import (
    ground_truth_answer,
    llm_structured,
    load_document_index,
)
from src.llm import get_model

# ---------------------------------------------------------------------------
# Structured judge output
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    """Structured judge output for answer quality (1-5)."""
    score: int
    explanation: str

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qa_pairs(qa_path):
    pairs = []
    with open(qa_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


# ---------------------------------------------------------------------------
# Retrieval accuracy (fast, no LLM needed)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM answer quality (slower, uses judge)
# ---------------------------------------------------------------------------

def llm_judge_score(
    client,
    question,
    generated_answer,
    ground_truth,
    model=None,
):
    model = model or get_model()
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
    # llm_structured patches the client (patch_openai_client) so text_format
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
    model = model or get_model()
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_comparison_chart(results, output_path):
    _, axes = plt.subplots(1, 3, figsize=(14, 5))

    simple = results["simple_rag"]
    agent = results["agentic_rag"]

    # 1. Retrieval accuracy (hit rate)
    ax = axes[0]
    methods = ["Simple RAG", "Agentic RAG"]
    hit_rates = [simple["retrieval"]["hit_rate"], agent["retrieval"]["hit_rate"]]
    colors = ["#4C72B0", "#55A868"]
    bars = ax.bar(methods, hit_rates, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Hit Rate (top-5)")
    ax.set_title("Retrieval Accuracy")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold")

    # 2. Average searches per query
    ax = axes[1]
    avg_searches = [1.0, agent["avg_searches_per_query"]]
    bars = ax.bar(methods, avg_searches, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Avg Searches / Query")
    ax.set_title("Search Overhead")
    for bar, val in zip(bars, avg_searches):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    # 3. Latency comparison
    ax = axes[2]
    latencies = [simple.get("latency_per_query", 0), agent.get("latency_per_query", 0)]
    bars = ax.bar(methods, latencies, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Latency (seconds/query)")
    ax.set_title("Latency Overhead")
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}s", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Agent Loop vs Simple RAG Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    from src.llm import get_api_key, get_base_url

    load_dotenv(PROJECT_ROOT / ".env")

    # Paths
    qa_path = PROJECT_ROOT / "evaluation" / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "agent_eval.json"
    chart_path = results_dir / "agent_eval_comparison.png"

    # Load Q&A pairs
    print("Loading Q&A pairs...")
    qa_pairs = load_qa_pairs(str(qa_path))
    print(f"Loaded {len(qa_pairs)} pairs")

    # Load document index — the source of ground-truth answers
    print("Loading document index (ground-truth answers)...")
    doc_idx = load_document_index()
    print(f"Loaded {len(doc_idx)} documents")

    base_url = get_base_url()
    api_key = get_api_key()

    llm_available = False
    client = None
    try:
        import urllib.request
        urllib.request.urlopen(base_url + "/models", timeout=3)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        llm_available = True
        print(f"LLM API available at {base_url}")
    except Exception:
        print(f"LLM API not available at {base_url} — running retrieval-only evaluation")

    # Initialize search
    print("Initializing search index...")
    t0 = time.time()
    from src.search.hybrid import HybridSearch

    search_index = HybridSearch()
    print(f"Search index ready in {time.time() - t0:.1f}s")

    # Initialize pipelines
    from src.rag.agent import RAGAgent
    from src.rag.pipeline import RAGBase

    rag_base = RAGBase(search_index=search_index, llm_client=client)
    agent = RAGAgent(search_index=search_index, llm_client=client)

    judge_sample = 20
    agent_full_sample = 50

    # -------------------------------------------------------------------------
    # Phase 1: Simple RAG evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: Simple RAG (single search)")
    print("=" * 60)

    # Retrieval accuracy (all dev-subset questions)
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

    # -------------------------------------------------------------------------
    # Phase 2: Agentic RAG evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: Agentic RAG (agent loop)")
    print("=" * 60)

    # Run agent on sample to measure searches and latency
    print("Running agent loop on sample...")
    agent_search_counts = []
    agent_latencies = []
    agent_retrieval_hits = 0
    agent_sample = qa_pairs[:judge_sample]

    for i, pair in enumerate(agent_sample):
        query = pair["question"]
        relevant_id = str(pair["document"])

        q_start = time.time()
        if llm_available:
            result = agent.run(query)
            q_elapsed = time.time() - q_start
            agent_search_counts.append(result["iterations"])
        else:
            search_results = rag_base.search(query)
            q_elapsed = time.time() - q_start
            agent_search_counts.append(1)
            result_searches = [type("SR", (), {"results": search_results})()]
            result = {"searches": result_searches, "answer": ""}

        agent_latencies.append(q_elapsed)

        all_retrieved_ids = []
        if llm_available:
            for search_record in result["searches"]:
                for doc in search_record.results:
                    all_retrieved_ids.append(str(doc.get("id", "")))
        else:
            for sr in result_searches:
                for doc in sr.results:
                    all_retrieved_ids.append(str(doc.get("id", "")))

        if relevant_id in all_retrieved_ids:
            agent_retrieval_hits += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(agent_sample)}] processed, avg searches: {np.mean(agent_search_counts):.2f}")

    avg_searches = np.mean(agent_search_counts)
    avg_latency = np.mean(agent_latencies)
    agent_retrieval_rate = agent_retrieval_hits / len(agent_sample)

    print(f"\n  Agent stats (n={len(agent_sample)}):")
    print(f"    Avg searches/query: {avg_searches:.2f}")
    print(f"    Avg latency/query: {avg_latency:.2f}s")
    print(f"    Retrieval hit rate: {agent_retrieval_rate:.1%}")

    if llm_available:
        print("\nEvaluating agent answer quality (LLM judge)...")
        agent_quality_scores = []
        agent_errors = 0

        for i, pair in enumerate(agent_sample):
            question = pair["question"]
            ground_truth = ground_truth_answer(doc_idx, pair["document"])
            if ground_truth is None:
                print(f"  [{i+1}/{len(agent_sample)}] No ground-truth document for question {pair.get('document')}")
                agent_errors += 1
                continue

            try:
                result = agent.run(question)
                generated = result["answer"]
            except Exception as e:
                print(f"  [{i+1}/{len(agent_sample)}] Error: {e}")
                agent_errors += 1
                continue

            judge_result = llm_judge_score(client, question, generated, ground_truth)
            if judge_result is None:
                agent_errors += 1
                continue

            agent_quality_scores.append(judge_result["score"])

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(agent_sample)}] evaluated")

        agent_mean_score = round(np.mean(agent_quality_scores), 2) if agent_quality_scores else 0

        print(f"  Mean score: {agent_mean_score}/5")
        print(f"  Evaluated: {len(agent_quality_scores)}, Errors: {agent_errors}")
    else:
        agent_quality_scores = []
        agent_mean_score = 0
        agent_errors = judge_sample

    # -------------------------------------------------------------------------
    # Phase 3: Full retrieval comparison (all 918)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3: Full retrieval comparison (dev subset)")
    print("=" * 60)

    full_sample = qa_pairs[:agent_full_sample]
    print(f"Running agent on {len(full_sample)} questions for retrieval metrics...")
    t0 = time.time()
    full_agent_search_counts = []
    full_agent_hits = 0

    for i, pair in enumerate(full_sample):
        query = pair["question"]
        relevant_id = str(pair["document"])

        if llm_available:
            result = agent.run(query)
            full_agent_search_counts.append(result["iterations"])
            all_ids = []
            for sr in result["searches"]:
                for doc in sr.results:
                    all_ids.append(str(doc.get("id", "")))
        else:
            search_results = rag_base.search(query)
            full_agent_search_counts.append(1)
            all_ids = [str(doc.get("id", "")) for doc in search_results]

        if relevant_id in all_ids:
            full_agent_hits += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(full_sample)}] processed")

    full_agent_time = time.time() - t0
    full_agent_retrieval_rate = full_agent_hits / len(full_sample)
    full_avg_searches = np.mean(full_agent_search_counts)

    print(f"  Agent full retrieval hit rate: {full_agent_retrieval_rate:.1%}")
    print(f"  Avg searches/query: {full_avg_searches:.2f}")
    print(f"  Total time: {full_agent_time:.1f}s")

    # -------------------------------------------------------------------------
    # Compile results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    results = {
        "simple_rag": {
            "accuracy": simple_retrieval["hit_rate"],
            "retrieval": {
                "hit_rate": simple_retrieval["hit_rate"],
                "hits": simple_retrieval["hits"],
                "total": simple_retrieval["total"],
            },
            "answer_quality": {
                "mean_score": simple_quality["mean_score"],
                "num_evaluated": simple_quality["num_evaluated"],
            },
            "latency_per_query": round(simple_latency_per_query, 2),
            "retrieval_time_seconds": round(simple_retrieval_time, 2),
        },
        "agentic_rag": {
            "accuracy": round(full_agent_retrieval_rate, 4),
            "retrieval": {
                "hit_rate": round(full_agent_retrieval_rate, 4),
                "hits": full_agent_hits,
                "total": len(full_sample),
            },
            "answer_quality": {
                "mean_score": agent_mean_score,
                "num_evaluated": len(agent_quality_scores),
            },
            "avg_searches_per_query": round(float(full_avg_searches), 2),
            "latency_per_query": round(float(avg_latency), 2),
            "total_time_seconds": round(full_agent_time, 2),
        },
        "comparison": {
            "retrieval_improvement": round(full_agent_retrieval_rate - simple_retrieval["hit_rate"], 4),
            "answer_quality_improvement": round(agent_mean_score - simple_quality["mean_score"], 2),
            "latency_overhead": round(float(avg_latency - simple_latency_per_query), 2),
            "search_overhead": round(float(full_avg_searches - 1.0), 2),
        },
        "config": {
            "total_questions": len(qa_pairs),
            "judge_sample_size": judge_sample,
            "model": get_model(),
        },
    }

    # Print summary
    r = results
    print(f"  Simple RAG retrieval hit rate:  {r['simple_rag']['retrieval']['hit_rate']:.1%}")
    print(f"  Agent RAG retrieval hit rate:   {r['agentic_rag']['retrieval']['hit_rate']:.1%}")
    print(f"  Retrieval improvement:          {r['comparison']['retrieval_improvement']:+.1%}")
    print(f"  Simple RAG answer score:        {r['simple_rag']['answer_quality']['mean_score']}/5")
    print(f"  Agent RAG answer score:         {r['agentic_rag']['answer_quality']['mean_score']}/5")
    print(f"  Answer quality improvement:     {r['comparison']['answer_quality_improvement']:+.2f}")
    print(f"  Avg searches/query (agent):     {r['agentic_rag']['avg_searches_per_query']}")
    print(f"  Latency overhead:               {r['comparison']['latency_overhead']:+.2f}s/query")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # Create visualization
    try:
        create_comparison_chart(results, str(chart_path))
    except Exception as e:
        print(f"Chart creation failed (non-fatal): {e}")

    return results


if __name__ == "__main__":
    main()

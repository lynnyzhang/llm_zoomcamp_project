import json
import sys
from pathlib import Path

from evaluation.answer_judge import JudgeScore
from src.llm import LLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from evaluation.agent_phases import (
        run_agentic_rag,
        run_retrieval_comparison,
        run_simple_rag,
    )
    from evaluation.agent_results import build_results, print_summary, write_results
    from evaluation.agent_setup import setup
    from evaluation.comparison_chart import create_comparison_chart

    qa_pairs, doc_idx, client, llm_available, rag_base, agent = setup()

    judge_sample = 20
    agent_full_sample = 50

    simple = run_simple_rag(rag_base, client, qa_pairs, doc_idx, judge_sample, llm_available)
    agentic = run_agentic_rag(agent, rag_base, client, qa_pairs, doc_idx, judge_sample, llm_available)
    comp = run_retrieval_comparison(agent, rag_base, qa_pairs, agent_full_sample, llm_available)

    results = build_results(qa_pairs, judge_sample, simple, agentic, comp)
    print_summary(results)

    output_path = PROJECT_ROOT / "evaluation" / "results" / "agent_eval.json"
    chart_path = PROJECT_ROOT / "evaluation" / "results" / "agent_eval_comparison.png"
    write_results(results, output_path)

    try:
        create_comparison_chart(results, str(chart_path))
    except Exception as e:
        print(f"Chart creation failed (non-fatal): {e}")

    return results


if __name__ == "__main__":
    main()

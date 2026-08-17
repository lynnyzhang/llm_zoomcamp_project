"""Collect LLM-judged gate data for notebooks 02/03/05: agent runs over a
strided dev-QA sample with per-gate full-doc/line scores, LLM-judged labels,
and quality scores, saved as one JSONL."""

import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.document_index import ground_truth_answer, load_document_index
from evaluation.judge_prompts import JUDGE_PROMPTS
from evaluation.llm_calls import llm_structured_retry
from evaluation.llm_judge import evaluate_single
from evaluation.notebooks.common import build_agent, load_qa, setup, trace_run
from src.llm_client import LLMClient
from src.rag.scoring import cosine_similarity

TYPE_KEYWORDS = {
    "stats": ["stat", "hp", "attack", "defense", "speed", "base"],
    "evolution": ["evolv", "baby", "stage", "chain"],
    "type": ["type", "weak", "resist", "super effective", "effectiveness"],
    "ability": ["abilit", "hidden", "overgrow", "chlorophyll"],
    "capture": ["capture"],
    "habitat": ["habitat"],
}


def classify(question):
    q = question.lower()
    for label, kws in TYPE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return label
    return "other"


class GroundingJudge(BaseModel):
    correctness: int
    grounded: bool
    explanation: str


def retrieved_texts(searches) -> list[str]:
    texts = []
    for rec in searches:
        for item in rec.results:
            texts.append(
                (
                    getattr(item, "search_text", "") or getattr(item, "snippet", "")
                ).strip()
            )
    return [t for t in texts if t]


def line_score(embedder, answer, texts):
    best = 0.0
    for t in texts:
        for line in t.split("\n"):
            line = line.strip()
            if line:
                best = max(best, cosine_similarity(embedder, answer, line))
    return best


def judge_gate(client, question, answer, texts, ground_truth):
    instructions = (
        "You judge a QA system. Rate correctness 1-5 against the ground truth "
        "(1 completely wrong, 5 fully correct), and state whether the answer is "
        "grounded in (supported by) the retrieved documents (boolean)."
    )
    prompt = (
        f"Question: {question}\n\nGenerated Answer: {answer}\n\n"
        f"Retrieved Documents:\n{chr(10).join(texts)}\n\n"
        f"Ground Truth: {ground_truth or 'N/A'}"
    )
    try:
        parsed, _ = llm_structured_retry(client, instructions, prompt, GroundingJudge)
    except Exception:
        return None, None, None
    return parsed.correctness, parsed.grounded, parsed.explanation


def judge_quality(client, question, texts, answer, ground_truth):
    context = "\n\n".join(texts)
    result = evaluate_single(
        client, question, context, answer, ground_truth, JUDGE_PROMPTS["with_examples"]
    )
    if result is None:
        return None
    return {
        "faithfulness": result["faithfulness"],
        "relevance": result["relevance"],
        "coherence": result["coherence"],
    }


def main():
    root = setup()
    agent, real_call_llm = build_agent()
    client = LLMClient.get().client
    qa = load_qa(root / "evaluation" / "data" / "qa.jsonl")
    sample = qa[::4][:60]
    doc_idx = load_document_index()
    print(
        f"sample: {len(sample)} questions over {len({q['document'] for q in sample})} distinct documents"
    )
    out_dir = root / "evaluation" / "notebooks" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate_collection.jsonl"
    out_path.write_text("", encoding="utf-8")
    judge_null = 0
    quality_null = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, item in enumerate(sample, 1):
            question = item["question"]
            document = item["document"]
            result, _calls, escalated, gate_history = trace_run(
                agent, question, real_call_llm
            )
            ground_truth = ground_truth_answer(doc_idx, document)
            gates = []
            for order, gate in enumerate(gate_history, 1):
                answer_to_judge = gate.rejected_answer or gate.answer or ""
                texts = retrieved_texts(gate.searches)
                judged = judge_gate(
                    client, question, answer_to_judge, texts, ground_truth
                )
                if judged[0] is None:
                    judge_null += 1
                gates.append(
                    {
                        "order": order,
                        "answer": answer_to_judge,
                        "full_doc_score": gate.confidence,
                        "line_score": line_score(
                            agent.embedder, answer_to_judge, texts
                        ),
                        "relevance": gate.relevance,
                        "judged_correctness": judged[0],
                        "judged_grounded": judged[1],
                        "judge_explanation": judged[2],
                    }
                )
            quality = None
            if not result.rejected:
                final_texts = retrieved_texts(result.searches)
                quality = judge_quality(
                    client, question, final_texts, result.answer, ground_truth
                )
                if quality is None:
                    quality_null += 1
            searches = [
                {
                    "query": rec.search_query or rec.query,
                    "texts": retrieved_texts([rec]),
                }
                for rec in result.searches
            ]
            row = {
                "id": i,
                "question": question,
                "document": document,
                "type": classify(question),
                "accepted": not result.rejected,
                "source": result.source,
                "escalated": escalated,
                "final_answer": result.answer,
                "final_confidence": result.confidence,
                "final_relevance": result.relevance,
                "searches": searches,
                "gates": gates,
                "quality": quality,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"[{i}/60] processed")
    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    n = len(rows)
    accepted = sum(1 for r in rows if r["accepted"])
    escalated_n = sum(1 for r in rows if r["escalated"])
    final_gates = [r["gates"][-1] for r in rows if r["gates"]]
    dist = Counter(
        g["judged_correctness"]
        for g in final_gates
        if g["judged_correctness"] is not None
    )
    print(f"\nsaved {n} rows -> {out_path}")
    print(f"n={n} accepted={accepted} rejected={n - accepted} escalated={escalated_n}")
    print("correctness distribution (final gate, judged):")
    for score in range(1, 6):
        print(f"  {score}: {dist.get(score, 0)}")
    print(f"judge nulls: {judge_null} | quality nulls: {quality_null}")


if __name__ == "__main__":
    main()

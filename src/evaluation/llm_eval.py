import json
import sys
import time
import types
from pathlib import Path

from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import get_model

# ---------------------------------------------------------------------------
# Pydantic model for structured judge output
# ---------------------------------------------------------------------------

class JudgeScores(BaseModel):
    faithfulness: int  # 1-5
    relevance: int     # 1-5
    coherence: int     # 1-5
    explanation: str   # brief reasoning


# ---------------------------------------------------------------------------
# Judge prompts (3 variants to compare)
# ---------------------------------------------------------------------------

JUDGE_PROMPTS = {
    "simple": {
        "instructions": "You are an AI judge. Rate the answer on a 1-5 scale.",
        "template": """\
Question: {question}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}

Rate faithfulness (context support), relevance (addresses question), coherence (clarity). 1-5 each.""",
    },
    "detailed": {
        "instructions": """\
You are an expert evaluator for question-answering systems.
Rate the generated answer on three dimensions using a 1-5 scale:
- Faithfulness (1-5): How well is the answer supported by the provided context? 5 = fully grounded, 1 = contradicts context.
- Relevance (1-5): Does the answer actually address the question asked? 5 = directly answers, 1 = off-topic.
- Coherence (1-5): Is the answer well-structured and clear? 5 = excellent, 1 = incoherent.
Provide a brief explanation for your ratings.""",
        "template": """\
Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Evaluate the generated answer against the context and ground truth.
Provide scores (1-5) for faithfulness, relevance, and coherence.""",
    },
    "with_examples": {
        "instructions": """\
You are an expert evaluator for question-answering systems.

Examples of ratings:

Example 1:
Q: "What type is Pikachu?"
Context: "Pikachu is an electric-type Pokémon."
Answer: "Pikachu is an electric type."
Ground Truth: "Pikachu is an electric type."
Rating: faithfulness=5, relevance=5, coherence=5

Example 2:
Q: "What type is Pikachu?"
Context: "Pikachu is an electric-type Pokémon."
Answer: "Pikachu is a grass type."
Ground Truth: "Pikachu is an electric type."
Rating: faithfulness=1, relevance=2, coherence=4

Now evaluate the following:""",
        "template": """\
Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Rate faithfulness (1-5), relevance (1-5), coherence (1-5). Explain briefly.""",
    },
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _patch_openai_client(client):
    # responses.parse sends a "text" field llama.cpp rejects — inject the
    # pydantic json_schema via extra_body.response_format instead.
    _original_parse = client.responses.parse

    def _patched_responses_parse(self, model, input, **kwargs):
        pydantic_model = kwargs.get("text_format", None)
        if not pydantic_model:
            return _original_parse(model=model, input=input, **kwargs)

        schema_name = getattr(pydantic_model, "__name__", "StructuredOutput")
        llama_cpp_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": pydantic_model.model_json_schema(),
            },
        }
        extra_body = kwargs.pop("extra_body", {})
        extra_body["response_format"] = llama_cpp_format
        return _original_parse(model=model, input=input, extra_body=extra_body, **kwargs)

    client.responses.parse = types.MethodType(_patched_responses_parse, client.responses)
    return client


def llm_judge(client, instructions, user_prompt, model=None):
    model = model or get_model()
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    # Try structured output first, fall back to text
    try:
        client = _patch_openai_client(client)
        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=JudgeScores,
        )
        return response.output_parsed
    except Exception:  # noqa: BLE001 — fall back to create-mode on any parse-mode failure
        # Fallback: use responses.create and parse JSON from text
        response = client.responses.create(
            model=model,
            input=messages,
        )
        text = response.output_text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            text = text.removeprefix("json")
            text = text.strip()
        return JudgeScores.model_validate_json(text)


def llm_judge_retry(client, instructions, user_prompt, model=None, max_retries=3):
    model = model or get_model()
    for attempt in range(max_retries):
        try:
            return llm_judge(client, instructions, user_prompt, model=model)
        except Exception as e:  # noqa: BLE001 — retry covers any judge failure
            if attempt == max_retries - 1:
                print(f"  Judge call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_qa_pairs(qa_path):
    pairs = []
    with open(qa_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def evaluate_single(
    client,
    question,
    context,
    generated_answer,
    ground_truth,
    judge_config,
    model=None,
):
    model = model or get_model()
    user_prompt = judge_config["template"].format(
        question=question,
        context=context,
        answer=generated_answer,
        ground_truth=ground_truth,
    )
    result = llm_judge_retry(client, judge_config["instructions"], user_prompt, model=model)
    if result is None:
        return None
    return {
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "coherence": result.coherence,
        "explanation": result.explanation,
    }


def evaluate_with_prompt(
    rag_pipeline,
    client,
    qa_pairs,
    judge_config,
    model=None,
    sample_size=50,
):
    model = model or get_model()
    pairs = qa_pairs[:sample_size] if sample_size > 0 else qa_pairs
    scores = []
    errors = 0

    for i, pair in enumerate(pairs):
        question = pair["question"]
        ground_truth = pair["answer"]

        # Generate answer via RAG pipeline
        try:
            search_results = rag_pipeline.search(question)
            context = rag_pipeline.build_context(search_results)
            generated = rag_pipeline.rag(question)
        except Exception as e:  # noqa: BLE001 — per-question errors never abort the batch
            print(f"  [{i+1}/{len(pairs)}] RAG error for question {pair.get('id')}: {e}")
            errors += 1
            continue

        # Judge the answer
        result = evaluate_single(
            client, question, context, generated, ground_truth, judge_config, model=model,
        )
        if result is None:
            errors += 1
            continue

        result["question_id"] = pair.get("id")
        result["question"] = question
        result["generated_answer"] = generated[:200]  # truncate for storage
        scores.append(result)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(pairs)}] processed")

    if not scores:
        return {"mean": {"faithfulness": 0, "relevance": 0, "coherence": 0}, "samples": [], "errors": errors}

    # Calculate means
    n = len(scores)
    mean = {
        "faithfulness": round(sum(s["faithfulness"] for s in scores) / n, 2),
        "relevance": round(sum(s["relevance"] for s in scores) / n, 2),
        "coherence": round(sum(s["coherence"] for s in scores) / n, 2),
    }

    return {
        "mean": mean,
        "samples": scores,
        "num_evaluated": n,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    from src.llm import get_api_key, get_base_url

    load_dotenv(PROJECT_ROOT / ".env")

    # Paths
    qa_path = PROJECT_ROOT / "data" / "qa.jsonl"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "llm_eval.json"

    # Load Q&A pairs
    print("Loading Q&A pairs...")
    qa_pairs = load_qa_pairs(str(qa_path))
    print(f"Loaded {len(qa_pairs)} pairs")

    base_url = get_base_url()
    api_key = get_api_key()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    print("Initializing RAG pipeline...")
    t0 = time.time()
    from src.rag.pipeline import RAGBase
    from src.search.hybrid import HybridSearch

    search_index = HybridSearch()
    rag = RAGBase(search_index=search_index, llm_client=client)
    print(f"RAG pipeline ready in {time.time() - t0:.1f}s")

    # Sample for evaluation (use 50 for speed, set to 0 for full)
    sample_size = 10

    # Evaluate with each judge prompt
    all_results = {}

    for name, config in JUDGE_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating with '{name}' judge prompt...")
        print(f"{'='*60}")

        t0 = time.time()
        result = evaluate_with_prompt(
            rag, client, qa_pairs, config,
            sample_size=sample_size,
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

    # Summary comparison
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

    # Determine best prompt by average score
    best_name = max(
        all_results.keys(),
        key=lambda k: (
            all_results[k]["mean"]["faithfulness"]
            + all_results[k]["mean"]["relevance"]
            + all_results[k]["mean"]["coherence"]
        ) / 3,
    )
    print(f"\nBest prompt: {best_name}")

    # Save results (strip per-sample details to keep file manageable)
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
            # Keep first 5 samples as examples
            "sample_examples": result.get("samples", [])[:5],
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

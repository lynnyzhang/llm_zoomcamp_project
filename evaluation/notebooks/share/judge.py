import time

from pydantic import BaseModel

from evaluation.notebooks.share.llm_calls import llm_structured, llm_structured_retry
from src.llm_client import LLMClient


class CorrectnessScore(BaseModel):
    score: int
    explanation: str


class QualityScore(BaseModel):
    faithfulness: int  # 1-5
    relevance: int  # 1-5
    coherence: int  # 1-5
    explanation: str  # brief reasoning


class GroundingJudge(BaseModel):
    correctness: int
    grounded: bool
    explanation: str


def llm_judge_score(client, question, generated_answer, ground_truth, model=None):
    model = model or LLMClient.get_model()
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
    try:
        parsed, _ = llm_structured(
            client, instructions, prompt, CorrectnessScore, model=model
        )
        if parsed is None:
            raise ValueError(
                "structured output unsupported: server returned output_parsed=None"
            )
        return {"score": parsed.score, "explanation": parsed.explanation}
    except Exception as e:
        print(f"  Judge error: {e}")
        return None


def llm_judge(client, instructions, user_prompt, model=None):
    model = model or LLMClient.get_model()
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=QualityScore,
    )
    if response.output_parsed is None:
        raise ValueError(
            "structured output unsupported: server returned output_parsed=None"
        )
    return response.output_parsed


def llm_judge_retry(client, instructions, user_prompt, model=None, max_retries=3):
    model = model or LLMClient.get_model()
    for attempt in range(max_retries):
        try:
            return llm_judge(client, instructions, user_prompt, model=model)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Judge call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2**attempt)


def evaluate_single(
    client,
    question,
    context,
    generated_answer,
    ground_truth,
    judge_config,
    model=None,
):
    model = model or LLMClient.get_model()
    user_prompt = judge_config["template"].format(
        question=question,
        context=context,
        answer=generated_answer,
        ground_truth=ground_truth,
    )
    result = llm_judge_retry(
        client, judge_config["instructions"], user_prompt, model=model
    )
    if result is None:
        return None
    return {
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "coherence": result.coherence,
        "explanation": result.explanation,
    }


def judge_grounding(client, question, answer, texts, ground_truth):
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

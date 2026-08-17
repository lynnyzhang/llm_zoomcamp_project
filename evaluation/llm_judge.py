import time

from pydantic import BaseModel

from src.llm_client import LLMClient


class JudgeScore(BaseModel):
    faithfulness: int  # 1-5
    relevance: int  # 1-5
    coherence: int  # 1-5
    explanation: str  # brief reasoning


def llm_judge(client, instructions, user_prompt, model=None):
    model = model or LLMClient.get_model()

    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    # Single path: structured output via responses.parse(text_format=JudgeScore).
    # The eval scripts use their own patch_openai_client (evaluation_utils)
    # for llama.cpp (response_format via extra_body), passing through to the
    # native SDK on OpenAI. There is deliberately NO
    # create()/JSON-extraction fallback: if the backend cannot honor the
    # schema, the output is prose — not JSON — so there is nothing to salvage
    # (llm_judge_retry retries transient failures).
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=JudgeScore,
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

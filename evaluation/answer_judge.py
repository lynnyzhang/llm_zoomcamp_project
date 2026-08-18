from pydantic import BaseModel

from evaluation.llm_calls import llm_structured
from src.llm_client import LLMClient


class JudgeScore(BaseModel):
    score: int
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

    # Single path: structured output via responses.parse(text_format=JudgeScore).
    # The eval scripts use their own patch_openai_client (evaluation_utils):
    # works on OpenAI (native structured output) AND on llama.cpp
    # (response_format via extra_body). There is deliberately NO create()/JSON
    # extraction fallback: if the backend cannot honor the schema, the output
    # is prose — not JSON — so there is nothing to salvage; fail loudly.
    try:
        parsed, _ = llm_structured(
            client, instructions, prompt, JudgeScore, model=model
        )
        if parsed is None:
            raise ValueError(
                "structured output unsupported: server returned output_parsed=None"
            )
        return {"score": parsed.score, "explanation": parsed.explanation}
    except Exception as e:
        print(f"  Judge error: {e}")
        return None

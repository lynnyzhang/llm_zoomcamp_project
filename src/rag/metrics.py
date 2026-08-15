from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    # Back-trace additions (production need): the row id and the fields the
    # UI/dashboard restore from the store.
    id: int | None = None
    question: str | None = None
    source: str | None = None
    rejected: bool = False
    span_id: str | None = None
    error: str | None = None

    @classmethod
    def failed(cls, model, answer, error, response_time):
        # Record for a failed turn: zero usage, no source, flagged rejected
        # so monitoring can count error turns without a real answer.
        return cls(
            model=model, prompt=None, instructions=None, answer=answer,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            response_time=response_time, cost=0.0,
            source=None, rejected=True, error=error,
        )

    @classmethod
    def from_response(cls, model, response, latency):
        # Per-call record from an LLM response: usage captured from
        # response.usage, cost from the token counts.
        call = cls(
            model=model, prompt=None, instructions=None, answer="",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            response_time=latency, cost=0.0,
        )
        raw = getattr(response, "usage", None)
        if raw is not None:
            # isinstance guard: test mocks return MagicMock usage whose
            # int() would raise.
            for key, attr in (("prompt_tokens", "input_tokens"),
                              ("completion_tokens", "output_tokens")):
                value = getattr(raw, attr, None)
                if isinstance(value, int):
                    setattr(call, key, value)
            call.total_tokens = call.prompt_tokens + call.completion_tokens
            call.cost = calculate_cost(
                model,
                {"input_tokens": call.prompt_tokens,
                 "output_tokens": call.completion_tokens},
            )
        return call

    @classmethod
    def call_failed(cls, model, latency):
        # Per-call record for an LLM call that raised: tokens stay None so
        # monitoring can distinguish failed calls from zero-usage ones.
        return cls(
            model=model, prompt=None, instructions=None, answer="",
            prompt_tokens=None, completion_tokens=None, total_tokens=None,
            response_time=latency, cost=0.0, error="LLM call failed",
        )

    @classmethod
    def turn(cls, model, result, usage, elapsed):
        # Turn-level record (the conversation row): aggregates per-call
        # usage, plus the result's source/rejection flags.
        return cls(
            model=model, prompt=None, instructions=None,
            answer=result.get("answer", ""),
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["input_tokens"] + usage["output_tokens"],
            response_time=elapsed,
            cost=calculate_cost(model, usage),
            source=result.get("source"),
            rejected=result.get("rejected", False),
        )


def calculate_cost(model, usage):
    cost = 0.0
    if "qwen" in model and usage:
        cost = (usage.get("input_tokens", 0) * 0.15 + usage.get("output_tokens", 0) * 0.60) / 1_000_000
    return cost

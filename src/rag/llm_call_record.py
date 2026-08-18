from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMCallSummary:
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency: float
    error: str | None

    @classmethod
    def from_record(cls, call):
        return cls(
            model=call.model,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            total_tokens=call.total_tokens,
            latency=call.response_time,
            error=call.error,
        )


@dataclass
class LLMCallRecord:
    model: str
    prompt: str | None
    instructions: str | None
    answer: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_time: float
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
        # Record for a failed agent loop: zero usage, no source, flagged
        # rejected so monitoring can count error loops without a real answer.
        return cls(
            model=model,
            prompt=None,
            instructions=None,
            answer=answer,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_time=response_time,
            source=None,
            rejected=True,
            error=error,
        )

    @classmethod
    def from_response(cls, model, response, latency):
        # Per-call record from an LLM response: usage captured from
        # response.usage.
        call = cls(
            model=model,
            prompt=None,
            instructions=None,
            answer="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_time=latency,
        )
        raw = getattr(response, "usage", None)
        if raw is not None:
            # isinstance guard: test mocks return MagicMock usage whose
            # int() would raise.
            for key, attr in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
            ):
                value = getattr(raw, attr, None)
                if isinstance(value, int):
                    setattr(call, key, value)
            if isinstance(call.prompt_tokens, int) and isinstance(
                call.completion_tokens, int
            ):
                call.total_tokens = call.prompt_tokens + call.completion_tokens
        return call

    @classmethod
    def call_failed(cls, model, latency):
        # Per-call record for an LLM call that raised: tokens stay None so
        # monitoring can distinguish failed calls from zero-usage ones.
        return cls(
            model=model,
            prompt=None,
            instructions=None,
            answer="",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            response_time=latency,
            error="LLM call failed",
        )

    @classmethod
    def agent_loop(cls, model, result, usage, elapsed):
        # Agent-loop record (the conversation row): aggregates per-call
        # usage, plus the result's source/rejection flags.
        return cls(
            model=model,
            prompt=None,
            instructions=None,
            answer=result.answer,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            response_time=elapsed,
            source=result.source,
            rejected=result.rejected,
        )

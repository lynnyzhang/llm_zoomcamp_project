"""Unit tests for the src.llm_client LLMClient.

All LLM calls are mocked — no real network, no real .env needed (conftest.py
sets MODEL_ID="test-model"). The process-wide singleton is reset between tests
via monkeypatch.setattr("src.llm_client.LLMClient.default_client", None).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.llm_client as llm


class Answer(BaseModel):
    text: str


def make_fake_openai(test_supported=True):
    """Return a fake OpenAI factory whose responses.parse is configured per
    test mode. test_supported=True → output_parsed non-None; False → the
    first call (the test) raises RuntimeError, later calls succeed."""
    fake = MagicMock()
    if test_supported:
        fake.responses.parse.return_value.output_parsed = object()
    else:
        fake.responses.parse.side_effect = [RuntimeError("unsupported"), MagicMock()]
    return fake


def reset_singleton(monkeypatch):
    monkeypatch.setattr("src.llm_client.LLMClient.default_client", None)


# ---------------------------------------------------------------------------
# LLMClient.get singleton
# ---------------------------------------------------------------------------

def test_get_returns_singleton(monkeypatch):
    reset_singleton(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "u")
    monkeypatch.setenv("MODEL_ID", "m")

    c1 = llm.LLMClient.get()
    c2 = llm.LLMClient.get()
    assert c1 is c2


# ---------------------------------------------------------------------------
# LLMClient construction / lazy client + text_format test/patch
# ---------------------------------------------------------------------------

def test_client_created_exactly_once(monkeypatch):
    fake = make_fake_openai(test_supported=True)
    monkeypatch.setattr("src.llm_client.OpenAI", lambda **kw: fake)

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    assert c.client is fake
    assert c.client is fake  # second access must not recreate
    assert fake.responses.parse.call_count == 1  # test ran exactly once


def test_text_format_and_patch_run_once(monkeypatch):
    fake = make_fake_openai(test_supported=False)
    monkeypatch.setattr("src.llm_client.OpenAI", lambda **kw: fake)
    original_parse = fake.responses.parse  # reference before patch

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    assert c.client is fake
    assert c.client is fake  # second access must not re-test/re-patch
    assert original_parse.call_count == 1  # test ran exactly once

    c.client.responses.parse(
        model="m", input=[{"role": "user", "content": "hi"}], text_format=Answer
    )
    call = original_parse.call_args
    kwargs = call.kwargs
    assert kwargs["text_format"] is Answer  # SDK keeps parsing -> output_parsed
    assert kwargs["extra_body"]["response_format"]["type"] == "json_schema"
    assert kwargs["extra_body"]["response_format"]["json_schema"]["name"] == "Answer"


# ---------------------------------------------------------------------------
# parse behavior (output_parsed, no manual JSON parsing)
# ---------------------------------------------------------------------------

def test_patched_parse_keeps_sdk_output_parsed(monkeypatch):
    fake = make_fake_openai(test_supported=False)
    monkeypatch.setattr("src.llm_client.OpenAI", lambda **kw: fake)
    response = MagicMock()
    response.output_parsed = Answer(text="parsed by the SDK")
    fake.responses.parse.side_effect = [RuntimeError("unsupported"), response]

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    result = c.client.responses.parse(
        model="m", input=[{"role": "user", "content": "hi"}], text_format=Answer
    )

    # text_format stays in the SDK call, so the SDK's own parser fills
    # output_parsed from the JSON reply — no manual parsing anywhere.
    assert result.output_parsed.text == "parsed by the SDK"


def test_parse_with_text_format_unsupported_injects_response_format(monkeypatch):
    fake = make_fake_openai(test_supported=False)
    monkeypatch.setattr("src.llm_client.OpenAI", lambda **kw: fake)
    original_parse = fake.responses.parse  # reference before patch

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    c.client.responses.parse(
        model="m", input=[{"role": "user", "content": "hi"}], text_format=Answer
    )

    call = original_parse.call_args
    kwargs = call.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert kwargs["text_format"] is Answer  # SDK keeps parsing -> output_parsed
    assert kwargs["extra_body"]["response_format"]["type"] == "json_schema"
    assert kwargs["extra_body"]["response_format"]["json_schema"]["name"] == "Answer"
    assert kwargs["extra_body"]["response_format"]["json_schema"]["strict"] is True


def test_parse_without_text_format_passes_through(monkeypatch):
    fake = make_fake_openai(test_supported=False)
    monkeypatch.setattr("src.llm_client.OpenAI", lambda **kw: fake)
    original_parse = fake.responses.parse  # reference before patch

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    c.client.responses.parse(model="m", input=[{"role": "user", "content": "hi"}])

    call = original_parse.call_args
    kwargs = call.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert "text_format" not in kwargs
    assert "extra_body" not in kwargs


def test_parse_with_text_format_supported_passes_through(monkeypatch):
    fake = make_fake_openai(test_supported=True)
    monkeypatch.setattr("src.llm_client.OpenAI", lambda **kw: fake)

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    c.client.responses.parse(
        model="m", input=[{"role": "user", "content": "hi"}], text_format=Answer
    )

    call = fake.responses.parse.call_args
    kwargs = call.kwargs
    assert kwargs["text_format"] is Answer
    assert "extra_body" not in kwargs


# ---------------------------------------------------------------------------
# env getters
# ---------------------------------------------------------------------------

def test_get_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm.LLMClient.get_api_key()


def test_get_base_url_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        llm.LLMClient.get_base_url()


def test_get_model_missing_raises(monkeypatch):
    monkeypatch.delenv("MODEL_ID", raising=False)
    with pytest.raises(RuntimeError):
        llm.LLMClient.get_model()


def test_temperature_getters_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_TEMPERATURE", raising=False)
    monkeypatch.delenv("ANSWER_TEMPERATURE", raising=False)
    assert llm.LLMClient.get_agent_temperature() == 0.0
    assert llm.LLMClient.get_answer_temperature() == 0.3


def test_temperature_getters_read_env(monkeypatch):
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.5")
    monkeypatch.setenv("ANSWER_TEMPERATURE", "0.7")
    assert llm.LLMClient.get_agent_temperature() == 0.5
    assert llm.LLMClient.get_answer_temperature() == 0.7


# ---------------------------------------------------------------------------
# explicit values without env
# ---------------------------------------------------------------------------

def test_llmclient_uses_explicit_values_without_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)

    c = llm.LLMClient(api_key="k", base_url="u", model="m")
    assert c.api_key == "k"
    assert c.base_url == "u"
    assert c.model == "m"

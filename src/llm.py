# Pure os.environ reader — no load_dotenv inside; callers own .env loading.
# The openai SDK only honors OPENAI_BASE_URL (not OPENAI_API_BASE_URL), so we
# build the client with an explicit base_url= pointing at the configured
# OpenAI-compatible endpoint (locally hosted LLM or cloud API).

import os

from openai import OpenAI
from pydantic import BaseModel

class Test(BaseModel):
    Id: int

client = None  # lazily created by create_client() and patched for llama.cpp support
def _env(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def get_api_key():
    key = _env("OPENAI_API_KEY")
    if key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return key


def get_base_url():
    url = _env("OPENAI_API_BASE_URL") or _env("OPENAI_BASE_URL")
    if url is None:
        raise RuntimeError(
            "OPENAI_API_BASE_URL is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return url


def get_model():
    model = _env("MODEL_ID")
    if model is None:
        raise RuntimeError(
            "MODEL_ID is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return model


def patch_openai_client(client):
    """
    Safely overrides client.responses.parse to support llama.cpp via
    'response_format' while falling back cleanly to the native SDK function
    when no text_format is requested.
    """
    _original_parse = client.responses.parse

    def _patched_responses_parse(self, model, input, **kwargs):
        pydantic_model = kwargs.pop("text_format", None)

        if pydantic_model is None:
            return _original_parse(model=model, input=input, **kwargs)

        llama_cpp_format = {
            "type": "json_schema",
            "json_schema": {
                "name": pydantic_model.__name__,
                "strict": True,
                "schema": pydantic_model.model_json_schema(),
            },
        }

        # Inject the formatting schema into the extra_body container payload
        extra_body = kwargs.pop("extra_body", {})
        extra_body["response_format"] = llama_cpp_format

        return _original_parse(
            model=model,
            input=input,
            extra_body=extra_body,
            **kwargs,
        )

    client.responses.parse = types.MethodType(_patched_responses_parse, client.responses)
    return client


def supports_text_format_parse(client, model, output_type):
    # Local servers (e.g. llama.cpp) accept the request but return plain text
    # (output_parsed=None), silently wasting a generation — probe once and patch if none.
    try:
        probe = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": 'Reply with JSON only: {"qa_pairs": []}'}],
            text_format=output_type
        )
        return probe.output_parsed is not None
    except Exception:  # noqa: BLE001 — any rejection means JSON fallback mode
        return False

def create_client():
    if client is None:
        client = OpenAI(api_key=get_api_key(), base_url=get_base_url())
        if not supports_text_format_parse(client, get_model(), Test):
            client = patch_openai_client(client)
    return client

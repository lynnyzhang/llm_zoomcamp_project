# Pure os.environ reader — no load_dotenv inside; callers own .env loading.
# The openai SDK only honors OPENAI_BASE_URL (not OPENAI_API_BASE_URL), so we
# build the client with an explicit base_url= pointing at the configured
# OpenAI-compatible endpoint (locally hosted LLM or cloud API).

import os

from openai import OpenAI


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


def create_client():
    return OpenAI(api_key=get_api_key(), base_url=get_base_url())

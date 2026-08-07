"""OpenAI client construction from environment.

Pure ``os.environ`` reader — no ``load_dotenv`` inside. Callers own
.env loading (e.g. eval scripts call ``load_dotenv(PROJECT_ROOT / ".env")``
at main() start, Streamlit loads it via dotenv on import).

The openai SDK only honors ``OPENAI_BASE_URL`` (not ``OPENAI_API_BASE_URL``),
so we build the client with an explicit ``base_url=`` pointing at the
configured OpenAI-compatible endpoint (locally hosted LLM or cloud API).
"""
import os
from openai import OpenAI


def _env(name: str) -> str | None:
    """Read ``name`` from env; return ``None`` for missing or empty/whitespace."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def get_api_key() -> str:
    key = _env("OPENAI_API_KEY")
    if key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return key


def get_base_url() -> str:
    url = _env("OPENAI_API_BASE_URL") or _env("OPENAI_BASE_URL")
    if url is None:
        raise RuntimeError(
            "OPENAI_API_BASE_URL is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return url


def get_model() -> str:
    model = _env("MODEL_ID")
    if model is None:
        raise RuntimeError(
            "MODEL_ID is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return model


def create_client() -> OpenAI:
    return OpenAI(api_key=get_api_key(), base_url=get_base_url())

# Pure os.environ reader — no load_dotenv inside; callers own .env loading.
# The openai SDK only honors OPENAI_BASE_URL (not OPENAI_API_BASE_URL), so we
# build the client with an explicit base_url= pointing at the configured
# OpenAI-compatible endpoint (locally hosted LLM or cloud API).
#
# LLMClient owns ALL retrieval: api key, base URL, and model are read from
# .env inside the class — the text_format
# test + llama.cpp patch run exactly once at first client access.

import os
import types

from openai import OpenAI
from pydantic import BaseModel


class Test(BaseModel):
    """Minimal output type for the one-time text_format support test."""
    qa_pairs: list


class LLMClient:
    """OpenAI-compatible client wrapper.

    Reads api_key / base_url / model from .env once (constructor), creates the
    underlying OpenAI client lazily exactly once, and tests + patches
    ``responses.parse`` text_format support exactly once at first client
    access. ``responses`` mirrors the underlying client so callers keep using
    ``llm_client.responses.create/parse``. Use ``LLMClient.get()`` for the
    process-wide singleton.
    """

    default_client = None  # process-wide singleton instance

    @staticmethod
    def env(name):
        value = os.environ.get(name)
        if value is None or not value.strip():
            return None
        return value

    @staticmethod
    def get_api_key():
        key = LLMClient.env("OPENAI_API_KEY")
        if key is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set or is empty in .env. Set it in project/.env (see .env.example)."
            )
        return key

    @staticmethod
    def get_base_url():
        url = LLMClient.env("OPENAI_API_BASE_URL") or LLMClient.env("OPENAI_BASE_URL")
        if url is None:
            raise RuntimeError(
                "OPENAI_API_BASE_URL is not set or is empty in .env. Set it in project/.env (see .env.example)."
            )
        return url

    @staticmethod
    def get_model():
        model = LLMClient.env("MODEL_ID")
        if model is None:
            raise RuntimeError(
                "MODEL_ID is not set or is empty in .env. Set it in project/.env (see .env.example)."
            )
        return model

    @classmethod
    def get(cls):
        """Return the process-wide LLMClient singleton — the underlying OpenAI
        client (and the text_format test/patch) is created exactly once."""
        if cls.default_client is None:
            cls.default_client = cls()
        return cls.default_client

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key if api_key is not None else self.get_api_key()
        self.base_url = base_url if base_url is not None else self.get_base_url()
        self.model = model if model is not None else self.get_model()
        self.openai_client = None
        self.text_format_supported = None

    @property
    def client(self):
        """The underlying OpenAI client — created lazily, exactly once."""
        if self.openai_client is None:
            self.openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.text_format_supported = self.test_text_format()
            if not self.text_format_supported:
                self.patch_parse()
        return self.openai_client

    @property
    def responses(self):
        """Mirror of the underlying client's responses API (create/parse)."""
        return self.client.responses

    def parse(self, model=None, input=None, text_format=None, **kwargs):
        """Structured-output parse with text_format support (patched once for
        llama.cpp-style servers that only accept response_format)."""
        return self.responses.parse(
            model=model, input=input, text_format=text_format, **kwargs
        )

    def test_text_format(self):
        # Local servers (e.g. llama.cpp) accept the request but return plain
        # text (output_parsed=None), silently wasting a generation — test
        # once at client creation and patch if unsupported.
        try:
            test = self.openai_client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": 'Reply with JSON only: {"qa_pairs": []}'}],
                text_format=Test,
            )
            return test.output_parsed is not None
        except Exception:  # noqa: BLE001 — any rejection means JSON fallback mode
            return False

    def patch_parse(self):
        """Safely overrides client.responses.parse to support llama.cpp via
        'response_format' while falling back cleanly to the native SDK function
        when no text_format is requested."""
        original_parse = self.openai_client.responses.parse

        def patched_responses_parse(self, model, input, **kwargs):
            pydantic_model = kwargs.pop("text_format", None)

            if pydantic_model is None:
                return original_parse(model=model, input=input, **kwargs)

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

            return original_parse(
                model=model,
                input=input,
                extra_body=extra_body,
                **kwargs,
            )

        self.openai_client.responses.parse = types.MethodType(
            patched_responses_parse, self.openai_client.responses
        )

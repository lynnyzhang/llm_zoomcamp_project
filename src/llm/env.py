import os


def env(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def get_api_key():
    key = env("OPENAI_API_KEY")
    if key is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return key


def get_base_url():
    url = env("OPENAI_API_BASE_URL") or env("OPENAI_BASE_URL")
    if url is None:
        raise RuntimeError(
            "OPENAI_API_BASE_URL is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return url


def get_model():
    model = env("MODEL_ID")
    if model is None:
        raise RuntimeError(
            "MODEL_ID is not set or is empty in .env. Set it in project/.env (see .env.example)."
        )
    return model


def get_agent_temperature():
    return float(os.environ.get("AGENT_TEMPERATURE", "0.0"))

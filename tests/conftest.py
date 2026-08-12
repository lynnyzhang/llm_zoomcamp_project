"""Test environment: MODEL_ID is required by src.llm.LLMClient.get_model() (no fallback)."""
import os

os.environ.setdefault("MODEL_ID", "test-model")

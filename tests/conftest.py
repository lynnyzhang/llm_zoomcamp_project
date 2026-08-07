"""Test environment: MODEL_ID is required by src.llm.get_model() (no fallback)."""
import os

os.environ.setdefault("MODEL_ID", "test-model")

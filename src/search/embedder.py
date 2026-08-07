"""
ONNX-based sentence embedder (all-MiniLM-L6-v2) without PyTorch.

Model files (tokenizer.json + model.onnx) are fetched from HuggingFace
via `src.data.download_model` (repo: Xenova/all-MiniLM-L6-v2).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

#: Default model directory: <project root>/models/Xenova/all-MiniLM-L6-v2
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Xenova" / "all-MiniLM-L6-v2"

#: Override the model location (used by Docker: EMBEDDER_MODEL_PATH=/app/data/models/...)
ENV_MODEL_PATH = "EMBEDDER_MODEL_PATH"

#: Cap per inference call. Large batches blow up onnxruntime attention memory
#: (e.g. 3200 x 128 seqlen needs ~2.5 GB) and trigger swap-thrashing.
BATCH_SIZE = 128


def default_model_dir() -> Path:
    """Resolve the model directory, honoring the EMBEDDER_MODEL_PATH env var."""
    override = os.environ.get(ENV_MODEL_PATH)
    return Path(override) if override else DEFAULT_MODEL_DIR


class Embedder:
    """Encode text into dense vectors using an ONNX MiniLM model.

    Matches sentence-transformers `all-MiniLM-L6-v2` output: mean pooling
    over the attention mask followed by L2 normalization.
    """

    def __init__(self, path: str | Path | None = None):
        """
        Args:
            path: Directory containing tokenizer.json and model.onnx.
                Defaults to `default_model_dir()`.
        """
        path = Path(path) if path is not None else default_model_dir()

        tokenizer_file = path / "tokenizer.json"
        model_file = path / "model.onnx"
        if not tokenizer_file.exists() or not model_file.exists():
            raise FileNotFoundError(
                f"ONNX embedding model not found in {path}. "
                "Run `uv run python -m src.data.download_model` "
                "to fetch tokenizer.json and model.onnx from Xenova/all-MiniLM-L6-v2."
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.session = ort.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text into a vector."""
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode a batch of texts into a (n, dim) matrix."""
        self.tokenizer.enable_padding()
        chunks = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        vectors = []
        for chunk in chunks:
            encoded = self.tokenizer.encode_batch(chunk)
            vectors.append(self._run(encoded, normalize=normalize))
        return np.concatenate(vectors, axis=0)

    def _run(self, encoded, normalize: bool) -> np.ndarray:
        """Run ONNX inference + mean pooling on one tokenized batch."""
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )

        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled

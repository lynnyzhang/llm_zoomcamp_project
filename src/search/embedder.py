import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Xenova" / "all-MiniLM-L6-v2"

# Override the model location (used by Docker: EMBEDDER_MODEL_PATH=/app/data/models/...)
ENV_MODEL_PATH = "EMBEDDER_MODEL_PATH"

# Cap per inference call. Large batches blow up onnxruntime attention memory
# (e.g. 3200 x 128 seqlen needs ~2.5 GB) and trigger swap-thrashing.
BATCH_SIZE = 128


def default_model_dir():
    override = os.environ.get(ENV_MODEL_PATH)
    return Path(override) if override else DEFAULT_MODEL_DIR


class Embedder:
    def __init__(self, path=None):
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

    def encode(self, text, normalize=True):
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        self.tokenizer.enable_padding()
        chunks = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        vectors = []
        for chunk in chunks:
            encoded = self.tokenizer.encode_batch(chunk)
            vectors.append(self.run(encoded, normalize=normalize))
        return np.concatenate(vectors, axis=0)

    def run(self, encoded, normalize):
        feed = {}
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

        # Mean pooling over the attention mask, then L2 normalize — matches
        # sentence-transformers all-MiniLM-L6-v2 output.
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled

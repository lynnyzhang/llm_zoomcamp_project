import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "Xenova" / "ms-marco-MiniLM-L-6-v2"
)

# Override the model location (used by Docker: RERANKER_MODEL_PATH=/app/data/models/...)
ENV_MODEL_PATH = "RERANKER_MODEL_PATH"


def default_model_dir():
    override = os.environ.get(ENV_MODEL_PATH)
    return Path(override) if override else DEFAULT_MODEL_DIR


class Reranker:
    # Cross-encoder re-ranker over the fused top-N results. The model is
    # optional: when it is not downloaded the pipeline falls back to the
    # RRF order unchanged, so search never breaks without it.
    def __init__(self):
        path = default_model_dir()
        self.tokenizer_file = path / "tokenizer.json"
        self.model_file = path / "model.onnx"
        self.available = self.tokenizer_file.exists() and self.model_file.exists()
        self._tokenizer: Tokenizer | None = None
        self._session: ort.InferenceSession | None = None

    def _load(self) -> tuple[Tokenizer, ort.InferenceSession]:
        if self._session is None:
            tokenizer = Tokenizer.from_file(str(self.tokenizer_file))
            tokenizer.enable_truncation(max_length=512)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            session = ort.InferenceSession(
                str(self.model_file), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = tokenizer
            self._session = session
        assert self._tokenizer is not None and self._session is not None
        return self._tokenizer, self._session

    def rerank(self, query, docs):
        if not self.available or not docs:
            return docs
        tokenizer, session = self._load()
        pairs = [
            (query, doc.get("search_text") or doc.get("name") or "") for doc in docs
        ]
        encoded = tokenizer.encode_batch(pairs)
        feed = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            ),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
        }
        logits = np.asarray(session.run(None, feed)[0])
        scores = 1.0 / (1.0 + np.exp(-logits))[:, 0]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]

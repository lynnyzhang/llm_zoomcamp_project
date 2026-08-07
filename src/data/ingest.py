"""
Data ingestion script for rag-mini-wikipedia dataset from HuggingFace.

Downloads:
- text-corpus: 3,200 passages (title + content)
- question-answer: 918 Q&A pairs

Output:
- data/corpus.jsonl
- data/qa.jsonl
"""

import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "rag-datasets/rag-mini-wikipedia"
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_corpus():
    """Load text-corpus subset from rag-mini-wikipedia."""
    ds = load_dataset(DATASET_NAME, "text-corpus", split="passages")
    return ds.to_list()


def load_qa():
    """Load question-answer subset from rag-mini-wikipedia."""
    ds = load_dataset(DATASET_NAME, "question-answer", split="test")
    return ds.to_list()


def save_jsonl(data: list[dict], filepath: Path) -> int:
    """Save list of dicts to JSONL file. Returns record count."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(data)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_NAME}...")

    # Download and save text-corpus
    print("  Loading text-corpus (3,200 passages)...")
    corpus = load_corpus()
    corpus_path = DATA_DIR / "corpus.jsonl"
    count = save_jsonl(corpus, corpus_path)
    print(f"  Saved {count} passages to {corpus_path}")

    # Download and save question-answer
    print("  Loading question-answer (918 Q&A pairs)...")
    qa = load_qa()
    qa_path = DATA_DIR / "qa.jsonl"
    count = save_jsonl(qa, qa_path)
    print(f"  Saved {count} Q&A pairs to {qa_path}")

    print("Done!")


if __name__ == "__main__":
    main()

import argparse
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

# Suppress HF telemetry and download progress noise.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

ONNX_CANDIDATES = [
    "onnx/model.onnx",
    "onnx/encoder_model.onnx",
    "model.onnx",
]


def download(repo="Xenova/all-MiniLM-L6-v2", dest="models"):
    dest = Path(dest) / repo
    dest.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id=repo)
    onnx_file = next((c for c in ONNX_CANDIDATES if c in files), None)
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")

    for remote, local in [
        ("tokenizer.json", "tokenizer.json"),
        (onnx_file, "model.onnx"),
    ]:
        src = hf_hub_download(repo_id=repo, filename=remote)
        dst = dest / local
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")

    onnx_ext = onnx_file + "_data"
    if onnx_ext in files:
        src = hf_hub_download(repo_id=repo, filename=onnx_ext)
        dst = dest / "model.onnx_data"
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  saved {dst}")
        else:
            print(f"  exists {dst}")

    return dest


def main():
    parser = argparse.ArgumentParser(description="Download the ONNX embedding model")
    parser.add_argument("--repo", default="Xenova/all-MiniLM-L6-v2", help="HF repo with ONNX model")
    parser.add_argument("--dest", default="models", help="Destination directory (default: <project>/models)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = project_root / dest

    model_dir = download(args.repo, dest)
    print(f"\nEmbedding model ready at: {model_dir}")


if __name__ == "__main__":
    main()

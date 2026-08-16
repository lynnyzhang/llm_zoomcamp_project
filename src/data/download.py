import io
import os
import zipfile
from pathlib import Path

import requests

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "patelris/pokemon-dataset-with-stats-and-types"
)

# Browser-like UA so Kaggle's static file endpoint serves the zip (no auth).
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATASET_PATH", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
POKEMON_CSV = RAW_DIR / "pokemon_complete.csv"
TYPES_CSV = RAW_DIR / "pokemon_types.csv"


def validate_archive_body(content, content_type=""):
    """Validate a downloaded body is a ZIP archive before we try to open it.

    Kaggle's anonymous download endpoint is now bot-blocked and returns an
    HTTP 200 HTML page (Google reCAPTCHA) instead of the zip, which would
    otherwise crash later with an unhelpful BadZipFile. Raise SystemExit with
    a clear, actionable FATAL message instead. Returns None on success.
    """
    if not content.startswith(b"PK"):
        lower = content.lower()
        if b"recaptcha" in lower:
            raise SystemExit(
                "FATAL: anonymous Kaggle dataset downloads are currently blocked "
                "by a bot check (reCAPTCHA page returned instead of the archive). "
                "Use the bundled raw CSVs already present in data/raw/ (they ship "
                "with the repo - see docs/setup.md), or set KAGGLE_USERNAME and "
                "KAGGLE_KEY env vars and retry."
            )
        raise SystemExit(
            f"FATAL: unexpected response type (got {content_type or 'unknown'}, "
            f"{len(content)} bytes); expected a ZIP archive."
        )


def download_archive(url=KAGGLE_DATASET_URL):
    # Must always use GET: the Kaggle API endpoint returns 404 for HEAD
    # requests, and the signed GCS URL is never hardcoded - always go through it.
    print(f"Downloading archive from {url} ...")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"FATAL: archive download failed: {exc}") from exc
    print(f"  HTTP {resp.status_code}, {len(resp.content)} bytes")
    validate_archive_body(resp.content, resp.headers.get("content-type", ""))
    return zipfile.ZipFile(io.BytesIO(resp.content))


def extract_raw_csvs():
    # Idempotent on the raw cache: documents.jsonl is derived from it and never
    # triggers a re-download, so the raw CSVs are the deterministic source.
    if POKEMON_CSV.exists() and TYPES_CSV.exists():
        print(f"Using cached raw CSVs: {POKEMON_CSV}, {TYPES_CSV}")
        return

    zf = download_archive()
    names = zf.namelist()

    wanted = {
        "pokemon_complete.csv": POKEMON_CSV,
        "pokemon_types.csv": TYPES_CSV,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        base = Path(name).name
        if base in wanted:
            RAW_DIR.joinpath(base).write_bytes(zf.read(name))
            print(f"Extracted {name} -> {RAW_DIR / base}")

    missing = [base for base in wanted if not wanted[base].exists()]
    if missing:
        raise SystemExit(
            f"FATAL: archive missing expected CSVs: {missing} (found: {names})"
        )

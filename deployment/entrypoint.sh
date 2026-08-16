#!/bin/bash
set -euo pipefail

# =============================================================================
# Entrypoint script for LLM Zoomcamp Capstone
# Orchestrates: data download → processing → indexing → monitoring → Streamlit
# =============================================================================

PROJECT_ROOT="/app"
DATA_DIR="${PROJECT_ROOT}/data"
CHUNKS_DIR="${DATA_DIR}/chunks"
POKEMON_FILE="${DATA_DIR}/pokemon.jsonl"
DOCUMENTS_FILE="${CHUNKS_DIR}/documents.jsonl"
MODELS_DIR="${DATA_DIR}/models"
EMBEDDER_MODEL_PATH="${MODELS_DIR}/Xenova/all-MiniLM-L6-v2"
export EMBEDDER_MODEL_PATH

# ---------------------------------------------------------------------------
# Reach a locally hosted LLM running on the Docker host (Mac/Windows/Linux with host-gateway). Cloud endpoints pass through unchanged.
# This file only ever executes inside the container, so the rewrite never
# applies to host-terminal runs. Cloud URLs (no localhost/127.0.0.1) pass
# through untouched.
# ---------------------------------------------------------------------------
if [[ -n "${OPENAI_API_BASE_URL:-}" ]]; then
    OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL//localhost/host.docker.internal}"
    OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL//127.0.0.1/host.docker.internal}"
    export OPENAI_API_BASE_URL
fi
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    OPENAI_BASE_URL="${OPENAI_BASE_URL//localhost/host.docker.internal}"
    OPENAI_BASE_URL="${OPENAI_BASE_URL//127.0.0.1/host.docker.internal}"
    export OPENAI_BASE_URL
fi

echo "============================================="
echo " LLM Zoomcamp Capstone - Starting Pipeline"
echo "============================================="

# Step 0: Seed the mounted data volume from the bundled CSVs (Kaggle anonymous
# downloads are bot-blocked). The compose mount (/app/data) is a named volume
# that masks image contents, so the seed lives at /app/data_seed/raw/ and is
# copied into the volume here.
echo ""
echo "[0/7] Seeding bundled Pokémon dataset..."
if [ ! -f "$DATA_DIR/raw/pokemon_complete.csv" ]; then
    if [ -f "/app/data_seed/raw/pokemon_complete.csv" ]; then
        mkdir -p "$DATA_DIR/raw"
        cp /app/data_seed/raw/pokemon_complete.csv "$DATA_DIR/raw/pokemon_complete.csv"
        cp /app/data_seed/raw/pokemon_types.csv "$DATA_DIR/raw/pokemon_types.csv"
        echo "  Seeded bundled CSVs into $DATA_DIR/raw/."
    else
        echo "  WARNING: bundled seed CSVs not found in /app/data_seed/raw/;"
        echo "  will try to download the dataset in the next step."
    fi
else
    echo "  Dataset already present, skipping seed."
fi

# Step 1: Download Pokémon dataset
echo ""
echo "[1/7] Downloading Pokémon dataset..."
if [ -f "$POKEMON_FILE" ]; then
    echo "  Dataset already exists, skipping download."
else
    uv run python -c "
from src.data.ingest import main
main()
"
    echo "  Dataset downloaded successfully."
fi

# Step 2: Process and chunk documents
echo ""
echo "[2/7] Processing and chunking documents..."
if [ -f "$DOCUMENTS_FILE" ]; then
    echo "  Chunks already exist, skipping processing."
else
    uv run python -c "
from src.data.chunker import main
main()
"
    echo "  Documents processed and chunked."
fi

# Step 3: Download ONNX embedding model
echo ""
echo "[3/7] Downloading ONNX embedding model..."
if [ -f "$EMBEDDER_MODEL_PATH/model.onnx" ] && [ -f "$EMBEDDER_MODEL_PATH/tokenizer.json" ]; then
    echo "  Model already exists, skipping download."
else
    uv run python -c "
from src.data.download_model import download
download('Xenova/all-MiniLM-L6-v2', '${MODELS_DIR}')
"
    echo "  Embedding model downloaded."
fi

# Step 4: Build search indices (pre-download model)
echo ""
echo "[4/7] Building search indices..."
uv run python -c "
from src.search.hybrid_search import HybridSearch
import os

# Check if we can build the index
documents_path = '${DOCUMENTS_FILE}'
if os.path.exists(documents_path):
    print('  Building hybrid search index...')
    hybrid = HybridSearch(documents_path=documents_path)
    print('  Index built successfully.')
else:
    print('  ERROR: No documents found, cannot build index.')
    exit(1)
"

# Step 5: Initialize monitoring database
echo ""
echo "[5/7] Initializing monitoring database..."
uv run python -c "
from monitoring.db_init import init_db, init_feedback
init_db()
init_feedback()
"

# Step 6: Launch Streamlit app
echo ""
echo "[6/7] Launching Streamlit app..."
echo "============================================="
echo " Pipeline complete! Starting Streamlit..."
echo "============================================="
echo ""

exec uv run streamlit run \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    src/interface/app.py \
    -- --show-confidence

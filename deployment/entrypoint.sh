#!/bin/bash
set -euo pipefail

# =============================================================================
# Entrypoint script for LLM Zoomcamp Capstone
# Orchestrates: data download → processing → indexing → monitoring → Streamlit
# =============================================================================

PROJECT_ROOT="/app"
DATA_DIR="${PROJECT_ROOT}/data"
CHUNKS_DIR="${DATA_DIR}/chunks"
CORPUS_FILE="${DATA_DIR}/corpus.jsonl"
DOCUMENTS_FILE="${CHUNKS_DIR}/documents.jsonl"
TRACES_DB="${PROJECT_ROOT}/monitoring/traces.db"
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

# Step 1: Download Pokémon dataset
echo ""
echo "[1/6] Downloading Pokémon dataset..."
if [ -f "$CORPUS_FILE" ]; then
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
echo "[2/6] Processing and chunking documents..."
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
echo "[3/6] Downloading ONNX embedding model..."
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
echo "[4/6] Building search indices..."
uv run python -c "
from src.search.hybrid import HybridSearch
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
echo "[5/6] Initializing monitoring database..."
mkdir -p "$DATA_DIR"
uv run python -c "
import sqlite3
from pathlib import Path

db_path = Path('${TRACES_DB}')
db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Create spans table if it doesn't exist (matching tracer.py schema)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS spans (
        name TEXT,
        start_time INTEGER,
        end_time INTEGER,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cost REAL,
        feedback TEXT,
        agent_iterations INTEGER,
        query TEXT,
        search_queries TEXT
    )
''')

conn.commit()
conn.close()
print('  Monitoring database initialized.')
"

# Step 6: Launch Streamlit app
echo ""
echo "[6/6] Launching Streamlit app..."
echo "============================================="
echo " Pipeline complete! Starting Streamlit..."
echo "============================================="
echo ""

exec uv run streamlit run src/interface/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

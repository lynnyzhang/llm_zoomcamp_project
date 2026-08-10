FROM python:3.13-slim

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code and configuration only.
# src/ and monitoring/ ship with the production tree; evaluation/ (offline,
# pre-deployment), data/, models/ and results/ are excluded.
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY .env.example ./.env.example

# Bundle the raw dataset CSVs (Kaggle anonymous downloads are bot-blocked).
COPY data/raw/pokemon_complete.csv data/raw/pokemon_types.csv /app/data_seed/raw/

COPY deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

# Health check: verify Streamlit is responding (python:3.13-slim has no curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# ============================================================
# BugTriage OpenEnv — Root Dockerfile
# Follows the official openenv-base multi-stage pattern with uv.
# Used by HuggingFace Spaces (Docker SDK) and the submission validator.
# ============================================================
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env

# Copy dependency manifests first for best layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies into a virtual environment via uv
RUN uv sync --frozen --no-install-project --no-editable && \
    uv sync --frozen --no-editable

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy all application source
COPY --from=builder /app/env /app/env

# Overwrite with actual source code
COPY models.py    /app/env/models.py
COPY client.py    /app/env/client.py
COPY __init__.py  /app/env/__init__.py
COPY app/         /app/env/app/
COPY server/      /app/env/server/
COPY inference.py /app/env/inference.py
COPY openenv.yaml /app/env/openenv.yaml

# Activate venv and set Python path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:${PYTHONPATH:-}"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

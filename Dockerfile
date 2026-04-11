# ============================================================
# BugTriage OpenEnv — Dockerfile
# Simple single-stage build for HuggingFace Spaces (Docker SDK).
# ============================================================
FROM python:3.11-slim

# System deps for uvicorn
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/env

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source
COPY models.py      /app/env/models.py
COPY client.py      /app/env/client.py
COPY __init__.py    /app/env/__init__.py
COPY app/           /app/env/app/
COPY server/        /app/env/server/
COPY inference.py   /app/env/inference.py
COPY openenv.yaml   /app/env/openenv.yaml
COPY README.md      /app/env/README.md

ENV PYTHONPATH="/app/env:${PYTHONPATH:-}"
ENV HOST="0.0.0.0"
ENV PORT="7860"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the HTTP server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

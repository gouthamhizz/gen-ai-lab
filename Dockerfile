# Multi-stage build for the FastAPI/Uvicorn service (Step 12).
FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch/sentence-transformers/crewai/ragas are heavy; a separate layer keeps rebuilds of app code fast.
RUN pip install -r requirements.txt

COPY src ./src
COPY .env.example .env.example
RUN mkdir -p /app/artifacts /app/data

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

# --workers scales with CPU cores; PROMETHEUS_MULTIPROC_DIR lets /metrics aggregate across them.
CMD ["sh", "-c", "uvicorn genai_platform.api.main:app --host 0.0.0.0 --port 8000 --workers ${API_WORKERS:-4}"]

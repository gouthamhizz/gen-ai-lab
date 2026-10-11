"""STEP 12 entrypoint. Run with:  uvicorn genai_platform.api.main:app --workers 4 --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from genai_platform.api.routes import router
from genai_platform.observability.metrics import MetricsMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="GenAI Platform", version="1.0.0",
             description="RAG + Agent system - FastAPI / Uvicorn / SSE token streaming (Step 12)")
app.add_middleware(MetricsMiddleware)
app.include_router(router)

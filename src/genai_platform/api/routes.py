"""STEP 12 - the REST API: FastAPI + Uvicorn + Server-Sent Events for token streaming.

Endpoints
  GET  /health            liveness + readiness (Ollama/Gemini, Qdrant, Redis)
  GET  /metrics           Prometheus exposition (Step 11)
  POST /ingest            run the Step 4 pipeline over a path, index into Qdrant
  POST /chat/stream       SSE: hybrid RAG (Steps 5-6) + Redis memory (Step 7), token-streamed
  POST /agent/stream      SSE: ReAct agent (Step 8) with thought/action/observation/token events
  POST /crew/report       CrewAI Researcher-Writer-Reviewer report (Step 9) - not streamed, minutes-long
  POST /eval/ragas        run RAGAS (Step 10) over the live pipeline, returns + saves the report

Every endpoint but /health and /metrics requires `X-API-Key`. Concurrency comes from FastAPI's async
handlers + multiple Uvicorn workers; the LLM gateway and Qdrant/Redis clients are safe to share across
requests (stdlib httpx/redis-py connection pooling).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Response
from sse_starlette.sse import EventSourceResponse

from genai_platform.agents.react import ReActAgent
from genai_platform.agents.tools import default_tools
from genai_platform.agents.crew import run_report
from genai_platform.api.deps import check_api_key, dense_retriever, hybrid_retriever, llm, memory, redis_client, retriever_for
from genai_platform.api.schemas import AgentRequest, ChatRequest, CrewRequest, EvalRequest, IngestRequest
from genai_platform.api.sse import guarded, sse
from genai_platform.config import get_settings
from genai_platform.ingestion.pipeline import ingest_path
from genai_platform.llm.gateway import LLMError, get_gateway
from genai_platform.observability.metrics import ACTIVE_STREAMS, AGENT_STEPS, RETRIEVAL_LAT, UNGROUNDED, TelemetryHandler
from genai_platform.observability.tracing import flush as flush_tracing
from genai_platform.observability.tracing import lc_config
from genai_platform.vectorstore.qdrant_store import get_store

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict:
    s = get_settings()
    checks: dict[str, bool] = {}
    try:
        checks["qdrant"] = bool(get_store().client.get_collections())
    except Exception:  # noqa: BLE001
        checks["qdrant"] = False
    try:
        checks["redis"] = memory().ping()
    except Exception:  # noqa: BLE001
        checks["redis"] = False
    try:
        get_gateway().primary  # constructed already means config parsed; a cheap check by design
        checks["llm_provider"] = s.llm_provider
    except Exception:  # noqa: BLE001
        checks["llm_provider"] = "unavailable"
    return {"status": "ok" if checks.get("qdrant") and checks.get("redis") else "degraded", "checks": checks}


@router.get("/metrics")
def metrics_endpoint() -> Response:
    from genai_platform.observability.metrics import render_metrics

    body, ctype = render_metrics()
    return Response(body, media_type=ctype)


@router.post("/ingest", dependencies=[Depends(check_api_key)])
def ingest(req: IngestRequest) -> dict:
    store = get_store(recreate=req.recreate_collection)
    stats = ingest_path(req.path, store, req.strategy, req.chunk_size, req.overlap)
    hybrid_retriever().refresh()  # keep the in-memory BM25 index in sync with what we just wrote
    return {"path": req.path, **stats}


@router.post("/chat/stream", dependencies=[Depends(check_api_key)])
async def chat_stream(req: ChatRequest):
    mem, pipeline = memory(), retriever_for(req.hybrid)
    from genai_platform.rag.pipeline import RagPipeline

    rag = RagPipeline(pipeline, llm())
    history = mem.render(req.session_id)
    from genai_platform.observability.metrics import TelemetryHandler
    telemetry = TelemetryHandler()
    cfg = lc_config(session_id=req.session_id, tags=["chat"], extra_callbacks=[telemetry])


    async def gen():
        ACTIVE_STREAMS.inc()
        t0 = time.perf_counter()
        try:
            hits = await asyncio.to_thread(rag.retrieve, req.message, req.k, req.source, cfg)
            RETRIEVAL_LAT.observe(time.perf_counter() - t0)
            yield sse("metrics", {"latency_s": round(time.perf_counter() - t0, 3), "n_sources": len(hits), **telemetry.summary()})
            answer = ""
            async for tok in rag.astream(req.message, hits, history, cfg):
                answer += tok
                yield sse("token", {"text": tok})
            from genai_platform.rag.prompts import validate_citations

            guard = validate_citations(answer, len(hits))
            if not guard["grounded"]:
                UNGROUNDED.inc()
            await asyncio.to_thread(mem.add_turn, req.session_id, "user", req.message)
            await asyncio.to_thread(mem.add_turn, req.session_id, "assistant", answer)
            yield sse("citations", guard)
            yield sse("metrics", {"latency_s": round(time.perf_counter() - t0, 3), "n_sources": len(hits)})
        finally:
            ACTIVE_STREAMS.dec()
            flush_tracing()

    return EventSourceResponse(guarded(gen()))


@router.post("/agent/stream", dependencies=[Depends(check_api_key)])
async def agent_stream(req: AgentRequest):
    mem = memory()
    agent = ReActAgent(get_gateway(), default_tools(dense_retriever()))
    history = mem.render(req.session_id)

    async def gen():
        ACTIVE_STREAMS.inc()
        final_answer, steps = "", 0
        try:
            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def produce():
                for ev in agent.run_stream(req.message, history):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
                loop.call_soon_threadsafe(q.put_nowait, None)

            loop.run_in_executor(None, produce)
            while True:
                ev = await q.get()
                if ev is None:
                    break
                if ev["type"] == "final":
                    final_answer, steps = ev["answer"], ev["steps"]
                    AGENT_STEPS.observe(steps)
                yield sse(ev["type"], ev)
            await asyncio.to_thread(mem.add_turn, req.session_id, "user", req.message)
            await asyncio.to_thread(mem.add_turn, req.session_id, "assistant", final_answer)
        finally:
            ACTIVE_STREAMS.dec()
            flush_tracing()

    return EventSourceResponse(guarded(gen()))


@router.post("/crew/report", dependencies=[Depends(check_api_key)])
async def crew_report(req: CrewRequest) -> dict:
    return await asyncio.to_thread(run_report, dense_retriever(), req.topic)


@router.post("/eval/ragas", dependencies=[Depends(check_api_key)])
async def eval_ragas(req: EvalRequest) -> dict:
    from genai_platform.evaluation.ragas_eval import run_ragas
    from genai_platform.rag.pipeline import RagPipeline

    pipeline = RagPipeline(dense_retriever(), llm())
    return await asyncio.to_thread(run_ragas, pipeline, None, req.k)

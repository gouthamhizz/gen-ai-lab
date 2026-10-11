"""STEP 11a - Prometheus metrics (what Grafana charts) + per-request telemetry callback.

Two complementary layers:
  * Langfuse  -> per-request TRACES (every chain step, prompt, completion, tokens, cost)      [tracing.py]
  * Prometheus-> fleet-wide AGGREGATES (rates, latency percentiles, token totals, active streams) [this file]

With several Uvicorn workers, prometheus_client needs multiprocess mode: set PROMETHEUS_MULTIPROC_DIR
(docker-compose does) and /metrics aggregates all worker processes.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest, multiprocess

_LAT = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
HTTP_REQS = Counter("http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_LAT = Histogram("http_request_duration_seconds", "HTTP latency (time to response headers for SSE)", ["route"], buckets=_LAT)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["kind"])                      # kind = prompt|completion
LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["status"])
LLM_LAT = Histogram("llm_call_duration_seconds", "LLM call duration", buckets=_LAT)
LLM_TTFT = Histogram("llm_time_to_first_token_seconds", "Time to first streamed token", buckets=_LAT)
RETRIEVAL_LAT = Histogram("retrieval_duration_seconds", "Retrieval (incl. query rewrite) duration", buckets=_LAT)
AGENT_STEPS = Histogram("agent_steps", "ReAct steps per run", buckets=(1, 2, 3, 4, 5, 6, 8))
UNGROUNDED = Counter("rag_ungrounded_answers_total", "Answers failing the citation guard")
ACTIVE_STREAMS = Gauge("sse_active_streams", "Open SSE streams", multiprocess_mode="livesum")


def render_metrics() -> tuple[bytes, str]:
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        return generate_latest(reg), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST


class MetricsMiddleware:
    """Pure-ASGI (BaseHTTPMiddleware buffers/breaks streaming responses, so we avoid it)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0, status = time.perf_counter(), 500

        async def send_wrap(msg):
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
                route = getattr(scope.get("route"), "path", scope["path"])
                HTTP_LAT.labels(route).observe(time.perf_counter() - t0)
            await send(msg)

        try:
            await self.app(scope, receive, send_wrap)
        finally:
            HTTP_REQS.labels(getattr(scope.get("route"), "path", scope["path"]), scope["method"], str(status)).inc()


class TelemetryHandler(BaseCallbackHandler):
    """LangChain callback: records latency, TTFT and token usage for every LLM call in a request.
    One instance per request -> `summary()` goes into the SSE `done` event ("token expenditure per API call")."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._start: dict[UUID, float] = {}
        self._first: dict[UUID, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kw: Any) -> None:
        self._start[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, **kw: Any) -> None:
        self._start[run_id] = time.perf_counter()

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kw: Any) -> None:
        if run_id not in self._first and run_id in self._start:
            self._first[run_id] = time.perf_counter() - self._start[run_id]
            LLM_TTFT.observe(self._first[run_id])

    def on_llm_end(self, response, *, run_id: UUID, **kw: Any) -> None:
        dur = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        p = c = 0
        for gen_list in response.generations:
            for g in gen_list:
                um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                p += um.get("input_tokens", 0)
                c += um.get("output_tokens", 0)
        LLM_TOKENS.labels("prompt").inc(p)
        LLM_TOKENS.labels("completion").inc(c)
        LLM_CALLS.labels("ok").inc()
        LLM_LAT.observe(dur)
        self.calls.append({"latency_s": round(dur, 3), "ttft_s": round(self._first.pop(run_id, dur), 3),
                           "prompt_tokens": p, "completion_tokens": c})

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        LLM_CALLS.labels("error").inc()
        self._start.pop(run_id, None)

    def summary(self) -> dict:
        return {"llm_calls": len(self.calls), "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
                "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
                "llm_latency_s": round(sum(c["latency_s"] for c in self.calls), 3),
                "ttft_s": self.calls[0]["ttft_s"] if self.calls else None}

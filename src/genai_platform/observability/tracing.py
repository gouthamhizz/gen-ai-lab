"""STEP 11b - Langfuse tracing for LCEL chains and agents.

Langfuse v3 SDK (OpenTelemetry based):
    from langfuse import get_client, observe
    from langfuse.langchain import CallbackHandler      # pass in `config={"callbacks": [...]}`
Session / user / tags travel as metadata keys `langfuse_session_id`, `langfuse_user_id`, `langfuse_tags`.

Everything degrades gracefully: without keys (or the package) these helpers are no-ops, so the app never
fails because observability is down.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable, Optional

from genai_platform.config import get_settings

log = logging.getLogger(__name__)


@lru_cache
def tracing_enabled() -> bool:
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return False
    try:
        import os

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
        from langfuse import get_client

        return bool(get_client().auth_check())
    except Exception as e:  # noqa: BLE001
        log.warning("Langfuse disabled: %s", e)
        return False


def lc_config(session_id: Optional[str] = None, user_id: Optional[str] = None, tags: Optional[list[str]] = None,
              extra_callbacks: Optional[list] = None, run_name: Optional[str] = None) -> dict:
    """Build a LangChain `config` dict: telemetry callbacks always, Langfuse handler when enabled."""
    callbacks = list(extra_callbacks or [])
    meta: dict[str, Any] = {}
    if tracing_enabled():
        from langfuse.langchain import CallbackHandler

        callbacks.append(CallbackHandler())
        if session_id:
            meta["langfuse_session_id"] = session_id
        if user_id:
            meta["langfuse_user_id"] = user_id
        if tags:
            meta["langfuse_tags"] = tags
    cfg: dict[str, Any] = {"callbacks": callbacks, "metadata": meta}
    if run_name:
        cfg["run_name"] = run_name
    return cfg


def observe_fn(name: str) -> Callable:
    """Decorator for NON-LangChain code (the hand-written ReAct loop, CrewAI run). No-op when disabled."""

    def deco(fn: Callable) -> Callable:
        if not tracing_enabled():
            return fn
        from langfuse import observe

        return observe(name=name)(fn)

    return deco


def flush() -> None:
    if tracing_enabled():
        from langfuse import get_client

        get_client().flush()

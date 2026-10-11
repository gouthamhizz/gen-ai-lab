"""STEP 1 - LLM API gateway.

One thin, dependable layer between our app and *any* LLM provider:

  * OllamaBackend        -> local Llama-3 8B over Ollama's REST API
  * GeminiBackend        -> Google's NEW `google-genai` SDK (`from google import genai`)
  * OpenAICompatBackend  -> the `openai` SDK pointed at Gemini's OpenAI-compatible
                            endpoint (or real OpenAI if you change OPENAI_BASE_URL)

Reliability features (this is the "robust wrapper" the step asks for):
  * exponential backoff with FULL JITTER  (delay = U(0, min(cap, base * 2**attempt)))
  * rate-limit parsing: honours `Retry-After`, `x-ratelimit-reset-*`, and Gemini's
    "retryDelay": "17s" / "Please retry in 17.5s" hints - the server's hint beats our guess
  * error classification: RateLimit / Transient (retry) vs Permanent (fail fast)
  * streaming retries ONLY before the first token (after that, retrying would duplicate text)
  * provider fallback: primary (Ollama) -> fallback (Gemini) when retries are exhausted

Every later step calls `get_gateway()`; nothing else in the repo talks to a provider.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Callable, Iterator, Mapping, Optional, Sequence

import httpx

from genai_platform.config import Settings, get_settings

log = logging.getLogger(__name__)
Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


# --------------------------------------------------------------------------- errors
class LLMError(Exception):
    """Base class for gateway errors."""


class RateLimitError(LLMError):
    def __init__(self, msg: str, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.retry_after = retry_after


class TransientError(LLMError):
    """Network blips, 5xx, timeouts - worth retrying."""


class PermanentError(LLMError):
    """Bad key, unknown model, invalid request - retrying cannot help."""


# --------------------------------------------------------------------------- results
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResult:
    text: str
    usage: Usage
    provider: str
    model: str
    latency_s: float


@dataclass
class Chunk:
    text: str = ""
    usage: Optional[Usage] = None  # set on the final chunk when the provider reports it


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) when a provider does not report usage."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- rate-limit parsing
_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")
_GEMINI_HINTS = [
    re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"'),
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I),
    re.compile(r"retry after (\d+(?:\.\d+)?)\s*(?:seconds|s)", re.I),
]


def parse_retry_after(headers: Optional[Mapping[str, str]] = None, message: str = "") -> Optional[float]:
    """Return seconds to wait, or None if the server gave no usable hint."""
    h = {k.lower(): v for k, v in (headers or {}).items()}
    if "retry-after-ms" in h:
        try:
            return float(h["retry-after-ms"]) / 1000
        except ValueError:
            pass
    if "retry-after" in h:
        v = h["retry-after"].strip()
        try:
            return float(v)  # delta-seconds form
        except ValueError:
            try:  # HTTP-date form
                return max(0.0, (parsedate_to_datetime(v) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):  # OpenAI style "1s", "6m0s", "20ms"
        if key in h:
            m = _DURATION.match(h[key].strip())
            if m and any(m.groups()):
                hrs, mins, secs, ms = (float(x) if x else 0.0 for x in m.groups())
                return hrs * 3600 + mins * 60 + secs + ms / 1000
    for rx in _GEMINI_HINTS:
        m = rx.search(message or "")
        if m:
            return float(m.group(1))
    return None


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Full-jitter exponential backoff (AWS Architecture Blog): U(0, min(cap, base*2^attempt))."""
    return random.uniform(0, min(cap, base * (2 ** attempt)))


@dataclass
class RetryConfig:
    max_retries: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    sleep: Callable[[float], None] = time.sleep  # injectable so tests do not really sleep


def _wait_time(exc: LLMError, attempt: int, cfg: RetryConfig) -> float:
    hinted = exc.retry_after if isinstance(exc, RateLimitError) else None
    if hinted is not None:
        return min(hinted + random.uniform(0, 0.25), cfg.max_delay)  # obey server, tiny jitter
    return backoff_delay(attempt, cfg.base_delay, cfg.max_delay)


def call_with_retry(fn: Callable[[], "LLMResult"], cfg: RetryConfig, stats: Counter, label: str = "") -> "LLMResult":
    for attempt in range(cfg.max_retries + 1):
        try:
            return fn()
        except (RateLimitError, TransientError) as exc:
            if attempt == cfg.max_retries:
                raise
            wait = _wait_time(exc, attempt, cfg)
            stats["retries"] += 1
            if isinstance(exc, RateLimitError):
                stats["rate_limited"] += 1
            log.warning("%s attempt %d failed (%s); sleeping %.2fs", label, attempt + 1, exc, wait)
            cfg.sleep(wait)
    raise AssertionError("unreachable")


def stream_with_retry(factory: Callable[[], Iterator[Chunk]], cfg: RetryConfig, stats: Counter, label: str = "") -> Iterator[Chunk]:
    """Retry only until the first chunk is yielded; mid-stream failures propagate."""
    for attempt in range(cfg.max_retries + 1):
        started = False
        try:
            for chunk in factory():
                started = True
                yield chunk
            return
        except (RateLimitError, TransientError) as exc:
            if started or attempt == cfg.max_retries:
                raise
            wait = _wait_time(exc, attempt, cfg)
            stats["retries"] += 1
            log.warning("%s stream attempt %d failed (%s); sleeping %.2fs", label, attempt + 1, exc, wait)
            cfg.sleep(wait)


# --------------------------------------------------------------------------- backends
class Backend(ABC):
    name: str
    model: str

    @abstractmethod
    def chat(self, messages: Sequence[Message], temperature: float, max_tokens: Optional[int], stop: Optional[list[str]]) -> LLMResult: ...

    @abstractmethod
    def stream(self, messages: Sequence[Message], temperature: float, max_tokens: Optional[int], stop: Optional[list[str]]) -> Iterator[Chunk]: ...


class OllamaBackend(Backend):
    """Local Llama-3 through Ollama's REST API (default http://localhost:11434)."""

    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 300.0):
        self.host, self.model = host.rstrip("/"), model
        self._timeout = httpx.Timeout(timeout, connect=5.0)

    def _payload(self, messages, temperature, max_tokens, stop, stream):
        options: dict = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop
        return {"model": self.model, "messages": list(messages), "stream": stream, "options": options}

    def _raise_for(self, resp: httpx.Response):
        if resp.status_code < 400:
            return
        body = resp.read().decode(errors="ignore")[:300] if hasattr(resp, "read") else ""
        if resp.status_code == 429:
            raise RateLimitError(body, parse_retry_after(resp.headers, body))
        if resp.status_code >= 500:
            raise TransientError(f"ollama {resp.status_code}: {body}")
        hint = f" (is the model pulled? run: ollama pull {self.model})" if resp.status_code == 404 else ""
        raise PermanentError(f"ollama {resp.status_code}: {body}{hint}")

    def chat(self, messages, temperature, max_tokens, stop) -> LLMResult:
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self._timeout) as c:
                with c.stream("POST", f"{self.host}/api/chat", json=self._payload(messages, temperature, max_tokens, stop, False)) as r:
                    self._raise_for(r)
                    data = json.loads(r.read())
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise TransientError(f"cannot reach Ollama at {self.host}: {e}") from e
        except (httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            raise TransientError(str(e)) from e
        text = data.get("message", {}).get("content", "")
        u = Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return LLMResult(text, u, self.name, self.model, time.perf_counter() - t0)

    def stream(self, messages, temperature, max_tokens, stop) -> Iterator[Chunk]:
        try:
            with httpx.Client(timeout=self._timeout) as c:
                with c.stream("POST", f"{self.host}/api/chat", json=self._payload(messages, temperature, max_tokens, stop, True)) as r:
                    self._raise_for(r)
                    for line in r.iter_lines():
                        if not line:
                            continue
                        d = json.loads(line)
                        if d.get("done"):
                            yield Chunk("", Usage(d.get("prompt_eval_count", 0), d.get("eval_count", 0)))
                        else:
                            yield Chunk(d.get("message", {}).get("content", ""))
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise TransientError(f"cannot reach Ollama at {self.host}: {e}") from e
        except (httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            raise TransientError(str(e)) from e


class GeminiBackend(Backend):
    """Gemini through the current SDK:  pip install google-genai ;  from google import genai."""

    name = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai  # imported lazily so Ollama-only users need no key

        if not api_key:
            raise PermanentError("GEMINI_API_KEY is empty")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def _cfg(self, messages, temperature, max_tokens, stop):
        from google.genai import types

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
            if m["role"] != "system"
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_tokens, stop_sequences=stop or None
        )
        return contents, cfg

    def _translate(self, exc: Exception) -> LLMError:
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            code, msg = getattr(exc, "code", 0) or 0, str(exc)
            if code == 429:
                return RateLimitError(msg, parse_retry_after(None, msg))
            if code >= 500 or code == 408:
                return TransientError(msg)
            return PermanentError(msg)
        if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
            return TransientError(str(exc))
        return PermanentError(str(exc))

    def chat(self, messages, temperature, max_tokens, stop) -> LLMResult:
        contents, cfg = self._cfg(messages, temperature, max_tokens, stop)
        t0 = time.perf_counter()
        try:
            r = self._client.models.generate_content(model=self.model, contents=contents, config=cfg)
        except Exception as e:  # noqa: BLE001 - translated into our taxonomy
            raise self._translate(e) from e
        um = r.usage_metadata
        usage = Usage(getattr(um, "prompt_token_count", 0) or 0, getattr(um, "candidates_token_count", 0) or 0)
        return LLMResult(r.text or "", usage, self.name, self.model, time.perf_counter() - t0)

    def stream(self, messages, temperature, max_tokens, stop) -> Iterator[Chunk]:
        contents, cfg = self._cfg(messages, temperature, max_tokens, stop)
        last_um = None
        try:
            for ch in self._client.models.generate_content_stream(model=self.model, contents=contents, config=cfg):
                last_um = ch.usage_metadata or last_um
                if ch.text:
                    yield Chunk(ch.text)
        except Exception as e:  # noqa: BLE001
            raise self._translate(e) from e
        if last_um:
            yield Chunk("", Usage(last_um.prompt_token_count or 0, last_um.candidates_token_count or 0))


class OpenAICompatBackend(Backend):
    """`openai` SDK -> Gemini's OpenAI-compatible endpoint (or OpenAI itself)."""

    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        if not api_key:
            raise PermanentError("OPENAI_API_KEY / GEMINI_API_KEY is empty")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)  # we own the retry policy

    def _translate(self, exc: Exception) -> LLMError:
        import openai

        if isinstance(exc, openai.RateLimitError):
            hdrs = getattr(getattr(exc, "response", None), "headers", None)
            return RateLimitError(str(exc), parse_retry_after(hdrs, str(exc)))
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)):
            return TransientError(str(exc))
        return PermanentError(str(exc))

    def chat(self, messages, temperature, max_tokens, stop) -> LLMResult:
        t0 = time.perf_counter()
        try:
            r = self._client.chat.completions.create(
                model=self.model, messages=list(messages), temperature=temperature, max_tokens=max_tokens, stop=stop
            )
        except Exception as e:  # noqa: BLE001
            raise self._translate(e) from e
        text = r.choices[0].message.content or ""
        u = Usage(getattr(r.usage, "prompt_tokens", 0) or 0, getattr(r.usage, "completion_tokens", 0) or 0)
        return LLMResult(text, u, self.name, self.model, time.perf_counter() - t0)

    def stream(self, messages, temperature, max_tokens, stop) -> Iterator[Chunk]:
        try:
            s = self._client.chat.completions.create(
                model=self.model, messages=list(messages), temperature=temperature, max_tokens=max_tokens, stop=stop, stream=True
            )
            for ev in s:
                if ev.choices and ev.choices[0].delta.content:
                    yield Chunk(ev.choices[0].delta.content)
        except Exception as e:  # noqa: BLE001
            raise self._translate(e) from e


# --------------------------------------------------------------------------- gateway
@dataclass
class LLMGateway:
    primary: Backend
    fallback: Optional[Backend] = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    stats: Counter = field(default_factory=Counter)

    def _candidates(self):
        yield self.primary
        if self.fallback is not None:
            yield self.fallback

    def chat(self, messages: Sequence[Message], temperature: float = 0.2, max_tokens: Optional[int] = None,
             stop: Optional[list[str]] = None) -> LLMResult:
        last: Optional[Exception] = None
        for be in self._candidates():
            try:
                res = call_with_retry(lambda be=be: be.chat(messages, temperature, max_tokens, stop), self.retry, self.stats, be.name)
                if res.usage.total == 0:  # provider gave no usage -> estimate so telemetry is never blank
                    res.usage = Usage(estimate_tokens(" ".join(m["content"] for m in messages)), estimate_tokens(res.text))
                self.stats[f"calls_{be.name}"] += 1
                return res
            except (RateLimitError, TransientError) as e:
                last = e
                self.stats["fallbacks"] += 1
                log.error("backend %s exhausted retries: %s", be.name, e)
        raise LLMError(f"all providers failed: {last}")

    def stream(self, messages: Sequence[Message], temperature: float = 0.2, max_tokens: Optional[int] = None,
               stop: Optional[list[str]] = None) -> Iterator[Chunk]:
        last: Optional[Exception] = None
        for be in self._candidates():
            started = False
            try:
                for ch in stream_with_retry(lambda be=be: be.stream(messages, temperature, max_tokens, stop), self.retry, self.stats, be.name):
                    started = True
                    yield ch
                self.stats[f"calls_{be.name}"] += 1
                return
            except (RateLimitError, TransientError) as e:
                if started:
                    raise
                last = e
                self.stats["fallbacks"] += 1
        raise LLMError(f"all providers failed: {last}")


def build_backend(kind: str, s: Settings) -> Backend:
    if kind == "ollama":
        return OllamaBackend(s.ollama_host, s.ollama_model)
    if kind == "gemini":
        return GeminiBackend(s.gemini_api_key, s.gemini_model)
    if kind == "openai":
        return OpenAICompatBackend(s.openai_api_key or s.gemini_api_key, s.openai_base_url, s.openai_model)
    raise ValueError(f"unknown LLM provider: {kind!r}")


@lru_cache
def get_gateway() -> LLMGateway:
    s = get_settings()
    primary = build_backend(s.llm_provider, s)
    fallback = None
    if s.llm_fallback_provider and s.llm_fallback_provider != s.llm_provider:
        try:
            fallback = build_backend(s.llm_fallback_provider, s)
        except LLMError as e:  # e.g. no Gemini key -> just run without a fallback
            log.warning("fallback disabled: %s", e)
    return LLMGateway(primary, fallback)

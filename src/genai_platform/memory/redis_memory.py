"""STEP 7 - Redis-backed conversation memory with rolling-summary compression.

Layout per session (all keys share a TTL so abandoned chats clean themselves up):
    chat:{sid}:turns        LIST of JSON {"role","content","ts"}   <- recent verbatim window
    chat:{sid}:summary      STRING  dense summary of everything older than the window
    chat:{sid}:summary_vec  STRING  JSON embedding of that summary (for similarity recall / analytics)

Compression policy: when the window exceeds `max_turns` messages OR `token_budget` estimated tokens, the OLDEST
half is folded into the summary by an LLM chain and trimmed from the list. Prompt size therefore stays bounded
no matter how long the chat runs:   prompt_history = summary (<=~120 words) + last few turns.
Honest note: the "dense context" is a text summary (LLMs consume text); we additionally store its embedding.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

from langchain_core.output_parsers import StrOutputParser

from genai_platform.llm.gateway import LLMError, estimate_tokens
from genai_platform.rag.prompts import SUMMARY_PROMPT

log = logging.getLogger(__name__)
Summarizer = Callable[[str, str], str]  # (previous_summary, rendered_old_turns) -> new_summary


def make_summarizer(llm) -> Summarizer:
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return lambda prev, turns: chain.invoke({"summary": prev or "(none)", "turns": turns}).strip()


class RedisMemory:
    def __init__(self, client, summarizer: Summarizer, embedder=None, max_turns: int = 10, token_budget: int = 1500,
                 ttl_s: int = 24 * 3600, lock_timeout_s: int = 5):
        self.r, self.summarize, self.embedder = client, summarizer, embedder
        self.max_turns, self.token_budget, self.ttl, self.lock_timeout = max_turns, token_budget, ttl_s, lock_timeout_s
        # NOTE: the compression lock uses redis-py`s Lock (SET NX + a Lua release script). fakeredis
        # has incomplete EVALSHA support, so unit tests should pass lock_timeout_s=0.2 and tolerate an
        # occasional skipped compression; integration tests against real Redis (docker-compose) are
        # unaffected.

    def _k(self, sid: str, part: str) -> str:
        return f"chat:{sid}:{part}"

    # ---- write ---------------------------------------------------------------
    def add_turn(self, sid: str, role: str, content: str) -> None:
        key = self._k(sid, "turns")
        self.r.rpush(key, json.dumps({"role": role, "content": content, "ts": time.time()}))
        self.r.expire(key, self.ttl)
        self._maybe_compress(sid)

    def _maybe_compress(self, sid: str) -> None:
        key = self._k(sid, "turns")
        raw = self.r.lrange(key, 0, -1)
        turns = [json.loads(x) for x in raw]
        tokens = sum(estimate_tokens(t["content"]) for t in turns)
        if len(turns) <= self.max_turns and tokens <= self.token_budget:
            return
        keep = max(2, self.max_turns // 2)
        old = turns[:-keep]
        if not old:
            return
        try:
            with self.r.lock(self._k(sid, "lock"), timeout=120, blocking_timeout=self.lock_timeout):  # one compressor per session
                prev = self.r.get(self._k(sid, "summary")) or ""
                rendered = "\n".join(f"{t['role']}: {t['content']}" for t in old)
                new = self.summarize(prev, rendered)
                pipe = self.r.pipeline()
                pipe.set(self._k(sid, "summary"), new, ex=self.ttl)
                pipe.ltrim(key, len(old), -1)  # drop exactly the turns we summarised (new ones may have arrived)
                if self.embedder is not None:
                    vec = self.embedder.embed_query(new)
                    pipe.set(self._k(sid, "summary_vec"), json.dumps([round(float(x), 5) for x in vec]), ex=self.ttl)
                pipe.execute()
        except LLMError as e:
            log.warning("memory compression skipped (LLM unavailable): %s", e)  # keep raw turns; retry next time
        except Exception as e:  # noqa: BLE001 - lock timeout etc. must not break the chat
            log.warning("memory compression skipped: %s", e)

    # ---- read ----------------------------------------------------------------
    def context(self, sid: str) -> tuple[str, list[dict]]:
        summary = self.r.get(self._k(sid, "summary")) or ""
        turns = [json.loads(x) for x in self.r.lrange(self._k(sid, "turns"), 0, -1)]
        return summary, turns

    def render(self, sid: str) -> str:
        summary, turns = self.context(sid)
        parts = []
        if summary:
            parts.append(f"[Summary of earlier conversation] {summary}")
        parts += [f"{t['role']}: {t['content']}" for t in turns]
        return "\n".join(parts)

    def clear(self, sid: str) -> None:
        self.r.delete(*(self._k(sid, p) for p in ("turns", "summary", "summary_vec")))

    def ping(self) -> bool:
        return bool(self.r.ping())

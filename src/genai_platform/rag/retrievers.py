"""STEP 5/6 - retrievers. All expose `retrieve(query, k, source=None) -> list[Hit]`."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

from genai_platform.llm.gateway import LLMError
from genai_platform.rag.prompts import MULTI_QUERY_PROMPT
from genai_platform.vectorstore.qdrant_store import Hit, QdrantStore

log = logging.getLogger(__name__)


class DenseRetriever:
    """Step 5: pure vector search."""

    def __init__(self, store: QdrantStore, embedder):
        self.store, self.embedder = store, embedder

    def retrieve(self, query: str, k: int = 5, source: Optional[str] = None) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(query), k, source)


# ---------------------------------------------------------------- Step 6 pieces
_STOP = set("a an the of to in on for and or is are was were be by with as at it this that from what which how".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOP]


class BM25Index:
    """Okapi BM25 over every chunk. score(q,d) = sum_t IDF(t) * f*(k1+1) / (f + k1*(1-b+b*|d|/avgdl)).
    Catches exact terms (error codes, names, acronyms) that dense embeddings blur."""

    def __init__(self, hits: list[Hit]):
        self.hits = hits
        self.bm25 = BM25Okapi([tokenize(h.text) for h in hits]) if hits else None

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(**{**self.hits[i].__dict__, "score": float(scores[i])}) for i in order if scores[i] > 0]


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum_over_lists 1/(k + rank). Needs no score calibration,
    which matters because cosine (0-1) and BM25 (0-30+) live on different scales."""
    fused: dict[str, float] = {}
    keep: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (k + rank)
            keep.setdefault(h.id, h)
    return [Hit(**{**keep[i].__dict__, "score": s}) for i, s in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


class QueryRewriter:
    """Multi-query: an LLM writes paraphrases so vocabulary mismatch ("car" vs "automobile") cannot hide a document."""

    def __init__(self, llm, n: int = 3):
        self.chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
        self.n = n

    def rewrite(self, question: str, config: Optional[dict] = None) -> list[str]:
        try:
            raw = self.chain.invoke({"question": question, "n": self.n}, config=config)
        except LLMError as e:  # retrieval must survive a flaky rewriter
            log.warning("query rewrite failed, using original only: %s", e)
            return [question]
        alts = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in raw.splitlines()]
        out, seen = [], set()
        for q in [question, *alts]:
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[: self.n + 1]


class HybridRetriever:
    """Step 6: {original + LLM rewrites} x {dense, BM25} -> RRF fusion -> top-k."""

    def __init__(self, store: QdrantStore, embedder, rewriter: Optional[QueryRewriter] = None, refresh_s: float = 30.0):
        self.store, self.embedder, self.rewriter, self.refresh_s = store, embedder, rewriter, refresh_s
        self._bm25: Optional[BM25Index] = None
        self._count, self._checked = -1, 0.0
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            self._bm25 = BM25Index(self.store.all_chunks())
            self._count, self._checked = self.store.count(), time.time()

    def _index(self) -> BM25Index:
        if self._bm25 is None or time.time() - self._checked > self.refresh_s:
            if self._bm25 is None or self.store.count() != self._count:
                self.refresh()
            else:
                self._checked = time.time()
        return self._bm25  # type: ignore[return-value]

    def retrieve(self, query: str, k: int = 5, source: Optional[str] = None, config: Optional[dict] = None,
                 dense_only: bool = False, bm25_only: bool = False) -> list[Hit]:
        queries = self.rewriter.rewrite(query, config) if self.rewriter else [query]
        rankings: list[list[Hit]] = []
        bm25 = self._index()
        for q in queries:
            if not bm25_only:
                rankings.append(self.store.search(self.embedder.embed_query(q), k * 2, source))
            if not dense_only:
                rankings.append([h for h in bm25.search(q, k * 3) if not source or h.source == source][: k * 2])
        return rrf(rankings)[:k]

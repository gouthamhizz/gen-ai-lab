"""STEP 4b - chunking: recursive character splitting and semantic chunking.

WHY CHUNK AT ALL: embeddings blur when a passage mixes topics, and LLM context is finite. Good
chunks are big enough to be self-contained and small enough to be about ONE thing.

recursive_split : try the coarsest separator first (paragraph -> line -> sentence -> word -> char)
                  and only descend when a piece is still too big. Preserves natural boundaries.
semantic_split  : embed every sentence, measure the cosine *distance* between neighbours and cut where
                  it spikes (topic shift). Percentile-based threshold adapts to each document.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from genai_platform.embeddings.math_ops import l2_normalize
from genai_platform.ingestion.schemas import Chunk, Document

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def recursive_split(text: str, chunk_size: int = 800, overlap: int = 100, separators: Optional[list[str]] = None) -> list[str]:
    seps = separators if separators is not None else _SEPARATORS
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    sep = next((s for s in seps if s == "" or s in text), "")
    rest = seps[seps.index(sep) + 1 :] if sep in seps else []
    if sep == "":
        parts = list(text)
    else:  # keep the separator attached so merged chunks read naturally
        raw = text.split(sep)
        parts = [r + sep for r in raw[:-1]] + [raw[-1]]
    # 1) recurse into oversize pieces, 2) greedily merge small pieces up to chunk_size
    pieces: list[str] = []
    for p in parts:
        pieces.extend(recursive_split(p, chunk_size, overlap, rest) if len(p) > chunk_size else [p])
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > chunk_size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            cur = tail[tail.find(" ") + 1 :] + p if " " in tail else p  # start overlap on a word boundary
        else:
            cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def semantic_split(text: str, embedder, percentile: float = 85.0, max_chars: int = 1200, min_sentences: int = 2) -> list[str]:
    sents = split_sentences(text)
    if len(sents) <= min_sentences:
        return [" ".join(sents)] if sents else []
    vecs = l2_normalize(embedder.embed_documents(sents))
    dist = 1.0 - (vecs[:-1] * vecs[1:]).sum(axis=1)          # cosine distance between neighbours
    cut_at = np.percentile(dist, percentile)
    chunks, cur = [], [sents[0]]
    for i, d in enumerate(dist):
        too_long = sum(len(s) for s in cur) + len(sents[i + 1]) > max_chars
        if (d > cut_at and len(cur) >= min_sentences) or too_long:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(sents[i + 1])
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_documents(docs: list[Document], strategy: str = "recursive", chunk_size: int = 800, overlap: int = 100, embedder=None) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        if strategy == "semantic":
            if embedder is None:
                raise ValueError("semantic chunking needs an embedder")
            pieces = semantic_split(doc.text, embedder, max_chars=chunk_size * 2)
        else:
            pieces = recursive_split(doc.text, chunk_size, overlap)
        for i, t in enumerate(pieces):
            out.append(Chunk(t, doc.source, doc.doc_type, doc.page, i, strategy))
    return out


def chunk_stats(chunks: list[Chunk]) -> dict:
    n = [len(c.text) for c in chunks] or [0]
    return {"chunks": len(chunks), "min": min(n), "mean": round(float(np.mean(n)), 1), "max": max(n)}

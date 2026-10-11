"""STEP 2b - text -> vector. One interface, three interchangeable backends.

  local  : Hugging Face sentence-transformers (default; free, offline, 384-d for MiniLM)
  gemini : `gemini-embedding-001` via google-genai; we request a smaller `output_dimensionality`
           (Matryoshka-trained) and L2-normalise, because truncated vectors are not unit length
  hash   : deterministic hashing bag-of-words; NO downloads - used by unit tests / CI only

Vectors are always returned L2-normalised float32, so cosine == dot everywhere downstream
(Qdrant's COSINE distance, our NumPy maths, BM25 fusion - all consistent).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

from genai_platform.config import get_settings
from genai_platform.embeddings.math_ops import l2_normalize
from genai_platform.llm.gateway import (PermanentError, RateLimitError, RetryConfig, TransientError,
                                        call_with_retry, parse_retry_after)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._m = SentenceTransformer(model_name)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self._m.encode(texts, batch_size=32, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, dim: int):
        from google import genai

        if not api_key:
            raise PermanentError("GEMINI_API_KEY is empty")
        self._c, self.name, self.dim = genai.Client(api_key=api_key), model, dim

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import errors, types

        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # API batch limit
            batch = texts[i : i + 100]

            def call(batch=batch):
                try:
                    return self._c.models.embed_content(model=self.name, contents=batch, config=cfg)
                except errors.APIError as e:
                    code = getattr(e, "code", 0) or 0
                    if code == 429:
                        raise RateLimitError(str(e), parse_retry_after(None, str(e))) from e
                    if code >= 500:
                        raise TransientError(str(e)) from e
                    raise PermanentError(str(e)) from e

            resp = call_with_retry(call, RetryConfig(), Counter(), "embed")
            out.extend(e.values for e in resp.embeddings)
        return l2_normalize(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class HashEmbedder:
    """Bag-of-words hashing trick. Not semantic, but deterministic and dependency-free."""

    name, dim = "hash-256", 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "local":
        return LocalEmbedder(s.embedding_model)
    if s.embedding_backend == "gemini":
        return GeminiEmbedder(s.gemini_api_key, s.gemini_embedding_model, s.gemini_embedding_dim)
    if s.embedding_backend == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBEDDING_BACKEND {s.embedding_backend!r}")

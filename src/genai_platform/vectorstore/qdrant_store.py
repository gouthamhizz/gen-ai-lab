"""STEP 3 - Qdrant vector store: schema, HNSW index, filtered similarity search.

Collection design
  vector   : `dim` floats, COSINE distance (vectors are already unit length)
  payload  : text, source, page, doc_type, chunk_id, index
  index    : HNSW graph (m=16, ef_construct=200)  +  keyword payload index on `source`
             so "search only inside handbook.pdf" is filtered *inside* the index, not after the fact

HNSW in one paragraph: a layered proximity graph. Search enters at the sparse top layer, greedily
walks toward the query, and drops down layers, so cost grows ~log(N) rather than N. `ef` trades
recall for speed at query time.

`QdrantClient(":memory:")` runs the same API in-process - used by the tests.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient, models

from genai_platform.ingestion.schemas import Chunk

_NS = uuid.UUID("6f1c2b7e-0000-4000-8000-000000000001")


@dataclass
class Hit:
    id: str
    score: float
    text: str
    source: str
    page: Optional[int] = None
    doc_type: str = ""
    chunk_id: str = ""

    @property
    def cite(self) -> str:
        return self.source + (f", p.{self.page}" if self.page else "")


def _hit(p) -> Hit:
    pl = p.payload or {}
    return Hit(str(p.id), float(getattr(p, "score", 0.0) or 0.0), pl.get("text", ""), pl.get("source", ""),
               pl.get("page"), pl.get("doc_type", ""), pl.get("chunk_id", ""))


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int, client: Optional[QdrantClient] = None):
        self.client = client or (QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url, timeout=30))
        self.collection, self.dim = collection, dim

    # ---- schema ------------------------------------------------------------------
    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            )
            self.client.create_payload_index(self.collection, "source", models.PayloadSchemaType.KEYWORD)

    # ---- write -------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 128) -> int:
        assert len(chunks) == len(vectors), "one vector per chunk"
        for i in range(0, len(chunks), batch):
            pts = [
                models.PointStruct(
                    id=str(uuid.uuid5(_NS, c.id)),  # deterministic -> idempotent re-ingestion
                    vector=v.tolist(),
                    payload={"text": c.text, "source": c.source, "page": c.page, "doc_type": c.doc_type, "chunk_id": c.id, "index": c.index},
                )
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch])
            ]
            self.client.upsert(self.collection, points=pts, wait=True)
        return len(chunks)

    # ---- read --------------------------------------------------------------------
    def search(self, vector: np.ndarray, k: int = 5, source: Optional[str] = None, ef: int = 128) -> list[Hit]:
        flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]) if source else None
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=k, query_filter=flt,
                                       with_payload=True, search_params=models.SearchParams(hnsw_ef=ef))
        return [_hit(p) for p in res.points]

    def timed_search(self, vector: np.ndarray, k: int = 5) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = self.search(vector, k)
        return hits, (time.perf_counter() - t0) * 1000  # ms, includes the network round-trip

    def all_chunks(self, limit: int = 100_000) -> list[Hit]:
        """Scroll the whole collection (used to build the in-memory BM25 index)."""
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True)
            out.extend(_hit(p) for p in pts)
            if offset is None or len(out) >= limit:
                return out

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def sources(self) -> list[str]:
        return sorted({h.source for h in self.all_chunks()})


def get_store(recreate: bool = False) -> QdrantStore:
    from genai_platform.config import get_settings
    from genai_platform.embeddings.embedder import get_embedder

    s = get_settings()
    dim = get_embedder().dim
    store = QdrantStore(s.qdrant_url, f"{s.qdrant_collection}_{dim}", dim)  # dim in the name: switching embedders never corrupts an index
    store.ensure_collection(recreate)
    return store

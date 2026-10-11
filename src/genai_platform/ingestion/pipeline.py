"""STEP 4c - glue: load -> chunk -> embed -> upsert. Produces the `chunks.jsonl` artifact."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from genai_platform.config import ARTIFACTS
from genai_platform.embeddings.embedder import get_embedder
from genai_platform.ingestion.chunking import chunk_documents, chunk_stats
from genai_platform.ingestion.loaders import load_path
from genai_platform.ingestion.schemas import Chunk


def build_chunks(path: str | Path, strategy: str = "recursive", chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    docs = load_path(path)
    return chunk_documents(docs, strategy, chunk_size, overlap, embedder=get_embedder() if strategy == "semantic" else None)


def save_chunks(chunks: list[Chunk], out: Optional[Path] = None) -> Path:
    out = out or ARTIFACTS / "chunks.jsonl"
    out.write_text("\n".join(json.dumps(c.to_dict()) for c in chunks), encoding="utf-8")
    return out


def load_chunks(path: Optional[Path] = None) -> list[Chunk]:
    path = path or ARTIFACTS / "chunks.jsonl"
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ingest_path(path: str | Path, store, strategy: str = "recursive", chunk_size: int = 800, overlap: int = 100) -> dict:
    chunks = build_chunks(path, strategy, chunk_size, overlap)
    if not chunks:
        return {"chunks": 0}
    vecs = get_embedder().embed_documents([c.text for c in chunks])
    store.upsert(chunks, vecs)
    return chunk_stats(chunks) | {"indexed": len(chunks)}

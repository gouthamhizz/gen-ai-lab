"""Plain data containers shared by ingestion, retrieval and the API."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Document:
    text: str
    source: str                      # file name, used for citations
    doc_type: str = "txt"            # pdf | html | txt | md
    page: Optional[int] = None       # 1-based page for PDFs
    title: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str = "txt"
    page: Optional[int] = None
    index: int = 0                   # position within its document
    strategy: str = "recursive"
    id: str = ""

    def __post_init__(self):
        if not self.id:  # deterministic id -> re-ingesting the same file overwrites instead of duplicating
            raw = f"{self.source}|{self.page}|{self.index}|{self.text[:64]}"
            self.id = hashlib.sha1(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cite(self) -> str:
        return f"{self.source}" + (f", p.{self.page}" if self.page else "")

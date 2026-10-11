"""STEP 5 (+6) - the RAG chain, written in LCEL.

    retrieve -> format numbered context -> RAG_PROMPT | llm | StrOutputParser -> citation guard

`retriever` is pluggable: Step 5 passes DenseRetriever, Step 6 passes HybridRetriever - same pipeline.
The chain is split from retrieval so the API can emit the `sources` SSE event BEFORE generation starts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from langchain_core.output_parsers import StrOutputParser

from genai_platform.rag.prompts import RAG_PROMPT, format_context, validate_citations


@dataclass
class RagAnswer:
    answer: str
    hits: list
    citations: dict
    timings: dict = field(default_factory=dict)

    def sources(self) -> list[dict]:
        return [{"n": i, "source": h.source, "page": h.page, "score": round(h.score, 4), "snippet": h.text[:200]}
                for i, h in enumerate(self.hits, start=1)]


class RagPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.chain = RAG_PROMPT | llm | StrOutputParser()

    def retrieve(self, question: str, k: int = 5, source: Optional[str] = None, config: Optional[dict] = None) -> list:
        if hasattr(self.retriever, "rewriter"):
            return self.retriever.retrieve(question, k, source, config=config)
        return self.retriever.retrieve(question, k, source)

    @staticmethod
    def _inputs(question: str, hits: list, history: str) -> dict[str, Any]:
        return {"question": question, "context": format_context(hits), "history": history or "(none)"}

    def answer(self, question: str, history: str = "", k: int = 5, source: Optional[str] = None,
               config: Optional[dict] = None) -> RagAnswer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k, source, config)
        t1 = time.perf_counter()
        text = self.chain.invoke(self._inputs(question, hits, history), config=config)
        return RagAnswer(text, hits, validate_citations(text, len(hits)),
                         {"retrieval_s": t1 - t0, "generation_s": time.perf_counter() - t1})

    async def astream(self, question: str, hits: list, history: str = "", config: Optional[dict] = None) -> AsyncIterator[str]:
        """Token stream for an already-retrieved context (API calls retrieve() first to emit sources)."""
        async for tok in self.chain.astream(self._inputs(question, hits, history), config=config):
            if tok:
                yield tok

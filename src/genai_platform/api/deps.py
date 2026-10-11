"""Shared, process-wide singletons + auth. Everything is built ONCE per worker and reused per request."""
from __future__ import annotations

from functools import lru_cache

import redis
from fastapi import Header, HTTPException

from genai_platform.config import get_settings
from genai_platform.embeddings.embedder import get_embedder
from genai_platform.llm.gateway import get_gateway
from genai_platform.llm.lc_model import GatewayChatModel
from genai_platform.memory.redis_memory import RedisMemory, make_summarizer
from genai_platform.rag.pipeline import RagPipeline
from genai_platform.rag.retrievers import DenseRetriever, HybridRetriever, QueryRewriter
from genai_platform.vectorstore.qdrant_store import get_store


@lru_cache
def llm():
    return GatewayChatModel(gateway=get_gateway())


@lru_cache
def dense_retriever() -> DenseRetriever:
    return DenseRetriever(get_store(), get_embedder())


@lru_cache
def hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(get_store(), get_embedder(), QueryRewriter(llm()))


def retriever_for(hybrid: bool):
    return hybrid_retriever() if hybrid else dense_retriever()


@lru_cache
def rag_pipeline(hybrid: bool = True) -> RagPipeline:
    return RagPipeline(retriever_for(hybrid), llm())


@lru_cache
def redis_client():
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def memory() -> RedisMemory:
    return RedisMemory(redis_client(), make_summarizer(llm()), get_embedder())


def check_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key not in get_settings().api_key_set:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return x_api_key

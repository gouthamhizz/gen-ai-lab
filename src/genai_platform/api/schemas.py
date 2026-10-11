"""Pydantic request/response models for the API (Step 12)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=128)
    k: int = Field(5, ge=1, le=20)
    source: Optional[str] = None
    hybrid: bool = True


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=128)


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory under the container's data/ mount")
    strategy: str = Field("recursive", pattern="^(recursive|semantic)$")
    chunk_size: int = Field(800, ge=100, le=4000)
    overlap: int = Field(100, ge=0, le=1000)
    recreate_collection: bool = False


class CrewRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)


class EvalRequest(BaseModel):
    k: int = Field(5, ge=1, le=20)

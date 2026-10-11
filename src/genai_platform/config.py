"""Central configuration (12-factor: everything comes from env vars / .env).

Every later step imports `get_settings()` instead of reading os.environ directly,
so the same code runs on your laptop, in Docker, and in the cloud.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"  # every step writes its output artifact here
ARTIFACTS.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # providers
    llm_provider: str = "ollama"
    llm_fallback_provider: str = ""
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    openai_api_key: str = ""
    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_model: str = "gemini-3.8-flash"

    # embeddings
    embedding_backend: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dim: int = 768

    # infra
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"
    redis_url: str = "redis://localhost:6380/0"

    # observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # api
    api_keys: str = "dev-key-change-me"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()

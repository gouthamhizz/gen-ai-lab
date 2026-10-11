import os, sys
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:1")  # deliberately unreachable in unit tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

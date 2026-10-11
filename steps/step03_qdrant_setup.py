"""STEP 3 demo: docker-compose brings up Qdrant on :6333; this script creates the collection,
indexes a few points, and times a similarity search. Run `docker compose --profile core up -d` first
(or set QDRANT_URL=:memory: to run without Docker for a quick check)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.ingestion.schemas import Chunk  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    store = get_store(recreate=True)
    emb = get_embedder()
    texts = ["Qdrant is a vector database written in Rust.", "HNSW is an approximate nearest-neighbour index.",
             "Bananas are a tropical fruit rich in potassium."]
    chunks = [Chunk(t, "demo.txt", index=i) for i, t in enumerate(texts)]
    store.upsert(chunks, emb.embed_documents(texts))
    print(f"Collection '{store.collection}' now has {store.count()} points.")

    hits, ms = store.timed_search(emb.embed_query("what vector index is fast for search?"), k=2)
    for h in hits:
        print(f"  score={h.score:.3f}  {h.text}")
    print(f"Search took {ms:.2f} ms")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "step03_qdrant_setup.json"
    out.write_text(json.dumps({"collection": store.collection, "count": store.count(), "search_ms": ms}, indent=2))
    print(f"Artifact written: {out}")

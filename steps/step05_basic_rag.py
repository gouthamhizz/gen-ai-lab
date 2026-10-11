"""STEP 5 demo: index step04's chunks into Qdrant, then run cited, dense-only RAG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.ingestion.pipeline import load_chunks  # noqa: E402
from genai_platform.llm.gateway import get_gateway  # noqa: E402
from genai_platform.llm.lc_model import GatewayChatModel  # noqa: E402
from genai_platform.rag.pipeline import RagPipeline  # noqa: E402
from genai_platform.rag.retrievers import DenseRetriever  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    chunks, emb, store = load_chunks(), get_embedder(), get_store(recreate=True)
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    print(f"Indexed {store.count()} chunks into '{store.collection}'.")

    rag = RagPipeline(DenseRetriever(store, emb), GatewayChatModel(gateway=get_gateway()))
    for q in ["What is the refund policy?", "What is the capital of Australia?"]:
        ans = rag.answer(q, k=3)
        print(f"\nQ: {q}\nA: {ans.answer}\nGrounded: {ans.citations['grounded']}  Sources: {[s['source'] for s in ans.sources()]}")

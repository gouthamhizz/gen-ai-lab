"""STEP 6 demo: multi-query rewriting + BM25/dense hybrid search via Reciprocal Rank Fusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.llm.gateway import get_gateway  # noqa: E402
from genai_platform.llm.lc_model import GatewayChatModel  # noqa: E402
from genai_platform.rag.pipeline import RagPipeline  # noqa: E402
from genai_platform.rag.retrievers import HybridRetriever, QueryRewriter  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    store, emb, llm = get_store(), get_embedder(), GatewayChatModel(gateway=get_gateway())
    retriever = HybridRetriever(store, emb, QueryRewriter(llm, n=3))
    retriever.refresh()
    rag = RagPipeline(retriever, llm)

    q = "how fast are refunds after purchase"  # deliberately uses different wording than the source doc
    rewrites = retriever.rewriter.rewrite(q)
    print("Rewrites:", rewrites)
    ans = rag.answer(q, k=4)
    print(f"\nQ: {q}\nA: {ans.answer}\nSources: {[s['source'] for s in ans.sources()]}")

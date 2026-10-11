"""STEP 9 demo: Researcher -> Writer -> Reviewer CrewAI pipeline over the ingested documents."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.agents.crew import run_report  # noqa: E402
from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.rag.retrievers import DenseRetriever  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    retriever = DenseRetriever(get_store(), get_embedder())
    result = run_report(retriever, "Company refund and warranty policy")
    print(result["report"])

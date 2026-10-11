"""STEP 10 demo: quantitative RAGAS evaluation of the live RAG pipeline."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.evaluation.ragas_eval import run_ragas  # noqa: E402
from genai_platform.llm.gateway import get_gateway  # noqa: E402
from genai_platform.llm.lc_model import GatewayChatModel  # noqa: E402
from genai_platform.rag.pipeline import RagPipeline  # noqa: E402
from genai_platform.rag.retrievers import DenseRetriever  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    rag = RagPipeline(DenseRetriever(get_store(), get_embedder()), GatewayChatModel(gateway=get_gateway()))
    report = run_ragas(rag)
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report saved to artifacts/ragas_report.json ({report['n_cases']} cases)")

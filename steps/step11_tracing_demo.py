"""STEP 11 demo: run the RAG chain with Langfuse tracing + Prometheus telemetry attached."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.llm.gateway import get_gateway  # noqa: E402
from genai_platform.llm.lc_model import GatewayChatModel  # noqa: E402
from genai_platform.observability.metrics import TelemetryHandler  # noqa: E402
from genai_platform.observability.tracing import flush, lc_config, tracing_enabled  # noqa: E402
from genai_platform.rag.pipeline import RagPipeline  # noqa: E402
from genai_platform.rag.retrievers import DenseRetriever  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    print(f"Langfuse tracing enabled: {tracing_enabled()}  (set LANGFUSE_* in .env and run `docker compose --profile full up -d`)")
    telemetry = TelemetryHandler()
    rag = RagPipeline(DenseRetriever(get_store(), get_embedder()), GatewayChatModel(gateway=get_gateway()))
    cfg = lc_config(session_id="step11-demo", tags=["demo"], extra_callbacks=[telemetry])
    ans = rag.answer("What is the warranty period?", k=3, config=cfg)
    print(ans.answer)
    print("\nPer-call telemetry:", telemetry.summary())
    flush()
    print("View traces at your Langfuse host (default http://localhost:3000) once LANGFUSE_* keys are set.")

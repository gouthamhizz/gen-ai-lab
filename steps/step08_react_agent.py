"""STEP 8 demo: a ReAct agent using Wikipedia, calculator and the RAG index as tools."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.agents.react import ReActAgent  # noqa: E402
from genai_platform.agents.tools import default_tools  # noqa: E402
from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.llm.gateway import get_gateway  # noqa: E402
from genai_platform.rag.retrievers import DenseRetriever  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    tools = default_tools(DenseRetriever(get_store(), get_embedder()))
    agent = ReActAgent(get_gateway(), tools)
    for ev in agent.run_stream("What is 12 squared, plus the number of moons of Jupiter?"):
        if ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] in ("thought", "action", "observation"):
            print(f"\n[{ev['type'].upper()}] {ev.get('tool', '')} {ev.get('input', '') or ev.get('text', '')}")
    print()

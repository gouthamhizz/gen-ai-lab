"""STEP 7 demo: two turns in the same session, showing history carried in the prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import redis  # noqa: E402

from genai_platform.config import get_settings  # noqa: E402
from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.llm.gateway import get_gateway  # noqa: E402
from genai_platform.llm.lc_model import GatewayChatModel  # noqa: E402
from genai_platform.memory.redis_memory import RedisMemory, make_summarizer  # noqa: E402
from genai_platform.rag.pipeline import RagPipeline  # noqa: E402
from genai_platform.rag.retrievers import DenseRetriever  # noqa: E402
from genai_platform.vectorstore.qdrant_store import get_store  # noqa: E402

if __name__ == "__main__":
    llm, emb = GatewayChatModel(gateway=get_gateway()), get_embedder()
    mem = RedisMemory(redis.Redis.from_url(get_settings().redis_url, decode_responses=True), make_summarizer(llm), emb)
    rag = RagPipeline(DenseRetriever(get_store(), emb), llm)
    sid = "step07-demo"
    mem.clear(sid)

    for q in ["What is the warranty period?", "And what about for cancellations?"]:
        history = mem.render(sid)
        ans = rag.answer(q, history=history, k=3)
        print(f"\nQ: {q}\nA: {ans.answer}")
        mem.add_turn(sid, "user", q)
        mem.add_turn(sid, "assistant", ans.answer)
    print("\nStored summary so far:", mem.context(sid)[0] or "(window not yet compressed)")

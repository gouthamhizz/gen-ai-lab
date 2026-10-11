"""STEP 1 demo: call the local Llama-3 8B through Ollama, with retry/backoff visible in the logs.
Requires Ollama running with your model pulled (`ollama pull llama3.1:8b`, or whatever OLLAMA_MODEL is)."""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO)

from genai_platform.llm.gateway import get_gateway  # noqa: E402

if __name__ == "__main__":
    gw = get_gateway()
    print(f"Primary backend: {gw.primary.name}/{gw.primary.model}"
          + (f"  |  fallback: {gw.fallback.name}/{gw.fallback.model}" if gw.fallback else "  |  no fallback configured"))
    result = gw.chat([{"role": "user", "content": "In one sentence, what is Retrieval-Augmented Generation?"}])
    print("\n--- non-streaming ---")
    print(result.text)
    print(f"(provider={result.provider} model={result.model} tokens={result.usage.total} latency={result.latency_s:.2f}s)")

    print("\n--- streaming ---")
    for chunk in gw.stream([{"role": "user", "content": "Count from 1 to 5, one number per line."}]):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n\nGateway stats:", dict(gw.stats))

    out = Path(__file__).resolve().parents[1] / "artifacts" / "step01_llm_gateway.json"
    out.write_text(json.dumps({"provider": result.provider, "model": result.model,
                               "tokens": result.usage.total, "stats": dict(gw.stats)}, indent=2))
    print(f"\nArtifact written: {out}")

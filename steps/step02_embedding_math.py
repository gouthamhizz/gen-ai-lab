"""STEP 2 demo: hand-written cosine/Euclidean vs. NumPy, then real sentence-transformers embeddings."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from genai_platform.embeddings.embedder import get_embedder  # noqa: E402
from genai_platform.embeddings.math_ops import cosine_matrix, euclid_matrix, top_k, verify_against_numpy  # noqa: E402

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    check = verify_against_numpy(a, b)
    print(f"Hand-written vs NumPy -> cosine_err={check['cosine_err']:.2e}  euclid_err={check['euclid_err']:.2e}  ok={check['ok']}")

    emb = get_embedder()
    print(f"\nEmbedder: {emb.name}  (dim={emb.dim})")
    docs = ["Qdrant is a vector database.", "Bananas are a tropical fruit.", "HNSW enables fast approximate search."]
    query = "which database is used for vectors?"
    doc_vecs, q_vec = emb.embed_documents(docs), emb.embed_query(query)
    cos = cosine_matrix(q_vec[None, :], doc_vecs)[0]
    euc = euclid_matrix(q_vec[None, :], doc_vecs)[0]
    for i in top_k(cos, 3):
        print(f"  cos={cos[i]:.3f}  euclid={euc[i]:.3f}  {docs[i]}")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "step02_embedding_math.json"
    out.write_text(json.dumps({"embedder": emb.name, "dim": emb.dim, "verification": check,
                               "ranking": [docs[i] for i in top_k(cos, 3)]}, indent=2))
    print(f"\nArtifact written: {out}")

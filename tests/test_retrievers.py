from genai_platform.rag.retrievers import BM25Index, rrf, tokenize
from genai_platform.vectorstore.qdrant_store import Hit


def test_tokenize_drops_stopwords_lowercases():
    assert tokenize("The Quick Brown Fox") == ["quick", "brown", "fox"]


def test_bm25_ranks_relevant_doc_first_with_realistic_corpus():
    texts = [
        "the qdrant vector database uses hnsw indexing for fast search",
        "bananas are yellow tropical fruit",
        "redis is an in memory key value store",
        "postgres is a relational sql database",
        "hnsw graphs enable approximate nearest neighbor search at scale",
    ]
    idx = BM25Index([Hit(str(i), 0, t, "s") for i, t in enumerate(texts)])
    hits = idx.search("vector database hnsw search")
    assert hits[0].text.startswith("the qdrant")


def test_rrf_fuses_rankings_without_score_calibration():
    a = [Hit("1", 0.9, "x", "s"), Hit("2", 0.8, "y", "s")]
    b = [Hit("2", 50.0, "y", "s"), Hit("3", 30.0, "z", "s")]
    fused = rrf([a, b])
    assert fused[0].id == "2"  # appears near the top of both rankings
    assert {h.id for h in fused} == {"1", "2", "3"}

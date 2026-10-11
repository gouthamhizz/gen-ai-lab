from genai_platform.ingestion.chunking import chunk_documents, recursive_split
from genai_platform.ingestion.schemas import Document


def test_recursive_split_respects_chunk_size():
    text = "This is a sentence. " * 200
    chunks = recursive_split(text, chunk_size=200, overlap=20)
    assert all(len(c) <= 260 for c in chunks)  # a little slack for overlap/merge
    assert "".join(chunks[:1])  # non-empty


def test_recursive_split_short_text_single_chunk():
    assert recursive_split("hello world", chunk_size=800) == ["hello world"]


def test_chunk_documents_assigns_stable_ids():
    doc = Document("Paragraph one.\n\nParagraph two is here.", "f.txt")
    c1 = chunk_documents([doc], "recursive", chunk_size=15, overlap=0)
    c2 = chunk_documents([doc], "recursive", chunk_size=15, overlap=0)
    assert [c.id for c in c1] == [c.id for c in c2]  # deterministic -> idempotent re-ingest


def test_chunk_documents_empty_text_yields_no_chunks():
    assert chunk_documents([Document("   ", "f.txt")]) == []

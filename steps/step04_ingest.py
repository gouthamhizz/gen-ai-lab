"""STEP 4 demo: load PDFs/HTML from ../data, chunk them, and save the chunks.jsonl artifact
(indexing into Qdrant happens in step05, so this step's output is inspectable on its own)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai_platform.ingestion.chunking import chunk_stats  # noqa: E402
from genai_platform.ingestion.pipeline import build_chunks, save_chunks  # noqa: E402

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    supported = {".pdf", ".html", ".htm", ".txt", ".md"}
    if not data_dir.exists() or not any(f.suffix.lower() in supported for f in data_dir.rglob("*")):
        print(f"Put your own PDFs/HTML in {data_dir} then re-run. Falling back to a tiny built-in sample.")
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "sample.html").write_text(
            "<html><head><title>Sample Policy</title></head><body><h1>Refunds</h1>"
            "<p>Cancellations made within 24 hours of purchase are fully refunded.</p>"
            "<h1>Warranty</h1><p>The standard product warranty is 12 months from the purchase date.</p></body></html>")

    chunks = build_chunks(data_dir, strategy="recursive", chunk_size=500, overlap=50)
    print(chunk_stats(chunks))
    out = save_chunks(chunks)
    print(f"Artifact written: {out}")

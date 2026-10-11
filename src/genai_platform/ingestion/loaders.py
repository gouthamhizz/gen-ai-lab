"""STEP 4a - document loaders for PDF and HTML (plus txt/md).

PDF : pypdf, one Document per page so citations can say "handbook.pdf, p.7".
HTML: BeautifulSoup; drop <script>/<style>/<nav>/<footer>, keep headings as markdown-style
      lines ("## Heading") so the chunker can respect section boundaries.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from genai_platform.ingestion.schemas import Document

log = logging.getLogger(__name__)
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)      # de-hyphenate PDF line wraps
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        txt = _clean(page.extract_text() or "")
        if txt:
            docs.append(Document(txt, path.name, "pdf", page=i, title=path.stem))
        else:
            log.warning("%s p.%d has no extractable text (scanned? run OCR first)", path.name, i)
    return docs


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(_NOISE_TAGS):
        t.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(h.name[1])
        h.replace_with("\n\n" + "#" * level + " " + h.get_text(" ", strip=True) + "\n\n")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for br in soup.find_all(["p", "div", "tr", "section", "article"]):
        br.append("\n\n")
    return _clean(soup.get_text()), title


def load_html(path: Path) -> list[Document]:
    text, title = html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [Document(text, path.name, "html", title=title or path.stem)] if text else []


def load_text(path: Path) -> list[Document]:
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    return [Document(text, path.name, path.suffix.lstrip("."), title=path.stem)] if text else []


_LOADERS = {".pdf": load_pdf, ".html": load_html, ".htm": load_html, ".txt": load_text, ".md": load_text}


def load_path(path: str | Path) -> list[Document]:
    """Load one file, or every supported file under a directory (recursively)."""
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in _LOADERS) if p.is_dir() else [p]
    docs: list[Document] = []
    for f in files:
        loader = _LOADERS.get(f.suffix.lower())
        if loader is None:
            raise ValueError(f"unsupported file type: {f.suffix}")
        try:
            docs.extend(loader(f))
        except Exception as exc:  # one corrupt file must not kill a whole batch
            log.error("failed to load %s: %s", f, exc)
    return docs

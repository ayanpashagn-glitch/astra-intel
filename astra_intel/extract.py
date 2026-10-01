"""Extract text per page and split it into page-bound chunks.

Chunks never cross a page boundary, so every citation is an exact page number.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

CHUNK_CHARS = 1000
OVERLAP_CHARS = 180
MIN_TOTAL_CHARS = 200  # below this a PDF is treated as empty / scanned


class ExtractionError(Exception):
    """Raised with a message that is safe to show to the user."""


@dataclass
class Chunk:
    page: int
    idx: int
    text: str


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line wraps
    text = re.sub(r"\[\s*(?:\d+|citation needed|note \d+)\s*\]", "", text, flags=re.I)  # [12]
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)  # URLs are noise for search and summaries
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _drop_running_headers(pages: list[str]) -> list[str]:
    """Remove lines repeated on >40% of pages (headers, footers, print stamps)."""
    if len(pages) < 4:
        return pages
    norm = lambda l: re.sub(r"\d+", "#", l.strip().lower())
    counts: Counter = Counter()
    for p in pages:
        counts.update({norm(l) for l in p.splitlines() if l.strip()})
    limit = 0.4 * len(pages)
    bad = {k for k, c in counts.items() if c > limit and len(k) > 3}
    return ["\n".join(l for l in p.splitlines() if norm(l) not in bad) for p in pages]


def extract_pages(data: bytes, filename: str) -> list[str]:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        text = _clean(text)
        # pseudo-pages of ~3000 chars so citations still point somewhere useful
        return [text[i : i + 3000] for i in range(0, len(text), 3000)] or [""]
    if not name.endswith(".pdf"):
        raise ExtractionError("Unsupported file type. Upload a PDF (or .txt / .md).")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ExtractionError("This PDF is password-protected.")
        pages = [_clean(p.extract_text() or "") for p in reader.pages]
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError, AttributeError, TypeError) as e:
        raise ExtractionError("This file could not be read — it looks corrupt or is not a valid PDF.") from e
    if not pages:
        raise ExtractionError("This PDF has no pages.")
    pages = _drop_running_headers(pages)
    if sum(len(p) for p in pages) < MIN_TOTAL_CHARS:
        raise ExtractionError(
            "No readable text found. This is probably a scanned/image-only PDF (OCR is not supported yet)."
        )
    return pages


def chunk_page(text: str, page: int, start_idx: int = 0) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n{2,}|(?<=[.!?])\n(?=[A-Z])", text) if p.strip()]
    flat = " ".join(" ".join(paras).split())
    if not flat:
        return []
    chunks, pos, idx = [], 0, start_idx
    while pos < len(flat):
        end = min(pos + CHUNK_CHARS, len(flat))
        if end < len(flat):  # prefer to break at a sentence end, then at a space
            cut = max(flat.rfind(". ", pos + CHUNK_CHARS // 2, end), flat.rfind("? ", pos + CHUNK_CHARS // 2, end))
            if cut == -1:
                cut = flat.rfind(" ", pos + CHUNK_CHARS // 2, end)
            end = cut + 1 if cut != -1 else end
        piece = flat[pos:end].strip()
        if len(piece) > 40:
            chunks.append(Chunk(page, idx, piece))
            idx += 1
        if end >= len(flat):
            break
        pos = max(end - OVERLAP_CHARS, pos + 1)
        sp = flat.find(" ", pos)  # start the next chunk on a word boundary, never mid-word
        if 0 <= sp < end:
            pos = sp + 1
    return chunks


_CITATION_MARK = re.compile(r"\b(Retrieved|Archived|ISBN|ISSN|doi|PMID|S2CID|Bibcode|arXiv)\b")


def is_reference_text(text: str, limit: int = 3) -> bool:
    """Bibliography / citation lists: real content words, but not something a user asks about."""
    return len(_CITATION_MARK.findall(text)) >= limit


def build_chunks(pages: list[str]) -> list[Chunk]:
    out: list[Chunk] = []
    for i, text in enumerate(pages, start=1):
        for ch in chunk_page(text, i, start_idx=len(out)):
            if not is_reference_text(ch.text):
                out.append(ch)
    for n, ch in enumerate(out):
        ch.idx = n
    return out

"""Upload pipeline: bytes -> pages -> chunks -> summary -> database."""
from __future__ import annotations

import hashlib

from . import db, qa
from .extract import ExtractionError, build_chunks, extract_pages
from .summarize import keywords, summarise


def ingest(data: bytes, filename: str, mode: str | None = "auto") -> dict:
    """Returns {'id', 'duplicate'}; raises ExtractionError with a user-safe message."""
    if not data:
        raise ExtractionError("The file is empty.")
    sha = hashlib.sha256(data).hexdigest()
    existing = db.find_by_sha(sha)
    if existing:
        return {"id": existing, "duplicate": True}
    pages = extract_pages(data, filename)
    chunks = build_chunks(pages)
    if not chunks:
        raise ExtractionError("No readable text found in this document.")
    summary, used_mode = summarise(pages, mode)
    doc_id = db.add_document(filename, sha, len(pages), chunks, summary, used_mode, keywords(pages))
    qa.invalidate()
    return {"id": doc_id, "duplicate": False}

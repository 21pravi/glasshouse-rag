"""
Loading and chunking the knowledge base.

Deliberately small: this module knows how to read documents off disk and split
them into overlapping windows. It knows nothing about retrieval, embeddings,
databases or models. Everything downstream is in `bm25.py` and `engine.py`.

(This was formerly `rag.py`, which also contained an OpenAI-embedding vector
store. That store became unreachable once both front-ends moved to the shared
BM25 engine, so it was removed rather than left to rot.)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100


def chunk_text(text: str, chunk_size: int | None = None,
               overlap: int | None = None) -> list[str]:
    """Split text into overlapping windows by character count.

    A character window rather than a sentence- or heading-aware splitter, so the
    Python and JavaScript implementations stay trivially identical — which is
    what the parity suite depends on.
    """
    chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = DEFAULT_OVERLAP if overlap is None else overlap
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def load_documents(directory: Path, chunk_size: int | None = None,
                   overlap: int | None = None) -> list[dict]:
    """Read every .md/.txt file in a directory and return its chunks.

    Each chunk carries its source filename and a content hash, so re-indexing an
    unchanged corpus is a no-op.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    docs: list[dict] = []
    for fpath in sorted(directory.glob("*")):
        if fpath.suffix.lower() not in (".md", ".txt"):
            continue
        text = fpath.read_text(encoding="utf-8")
        for chunk in chunk_text(text, chunk_size, overlap):
            docs.append({
                "source": fpath.name,
                "text": chunk,
                "hash": hashlib.md5(f"{fpath.name}:{chunk}".encode()).hexdigest(),
            })
    return docs

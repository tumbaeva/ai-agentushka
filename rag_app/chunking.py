"""Heading-aware chunking.

Instead of blindly splitting on a fixed character count, we walk the document
structure: markdown headings (# .. ######) start a new section, paragraphs are
the atomic units. A paragraph that is itself too long is split on sentence
boundaries. Every chunk keeps ``section`` metadata so citations can point at
the right part of the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .loaders import LoadedDoc

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    file_name: str
    section: str
    doc_type: str
    category: str
    chunk_index: int
    chars: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str, limit: int, overlap: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            buf = sent
        elif len(buf) + 1 + len(sent) <= limit:
            buf += " " + sent
        else:
            pieces.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + " " + sent) if tail else sent
    if buf:
        pieces.append(buf)
    return [p.strip() for p in pieces if p.strip()]


def chunk_document(doc: LoadedDoc, chunk_size: int = 700, chunk_overlap: int = 120,
                   min_chunk_size: int = 120, start_id: int = 0) -> list[Chunk]:
    current_section = doc.name
    chunks: list[Chunk] = []
    index = 0
    buffer: list[str] = []
    buffer_len = 0

    def make(text: str) -> Chunk:
        return Chunk(
            chunk_id=start_id + index,
            text=text,
            file_name=doc.name,
            section=current_section,
            doc_type=doc.doc_type,
            category=doc.category,
            chunk_index=index,
            chars=len(text),
        )

    def emit(text: str) -> None:
        nonlocal index
        text = text.strip()
        if not text:
            return
        chunks.append(make(text))
        index += 1

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer = []
        buffer_len = 0
        if not text:
            return
        if len(text) <= chunk_size:
            emit(text)
            return
        for piece in _split_long(text, chunk_size, chunk_overlap):
            emit(piece)

    for raw_line in doc.text.split("\n"):
        stripped = raw_line.strip()
        heading = HEADING_RE.match(stripped)
        if heading:
            flush()
            current_section = heading.group(2).strip() or current_section
            buffer.append(stripped)
            buffer_len = len(stripped)
            continue
        if not stripped:
            continue
        buffer.append(stripped)
        buffer_len += len(stripped) + 1
        if buffer_len >= chunk_size:
            flush()

    flush()
    return chunks


def chunk_documents(docs: list[LoadedDoc], chunk_size: int = 700, chunk_overlap: int = 120,
                    min_chunk_size: int = 120) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    next_id = 0
    for doc in docs:
        doc_chunks = chunk_document(
            doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size, start_id=next_id,
        )
        all_chunks.extend(doc_chunks)
        next_id += len(doc_chunks)
    return all_chunks

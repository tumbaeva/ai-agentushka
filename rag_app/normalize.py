"""Text normalization and tokenization helpers."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04FF0-9_\-]+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace / non-breaking spaces, trim."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens used by BM25 and overlap checks."""
    return _TOKEN_RE.findall(text.lower())


def shorten(text: str, width: int = 200) -> str:
    text = text.strip()
    if len(text) <= width:
        return text
    return text[:width].rstrip() + "..."


def highlight_fragment(text: str, query: str, radius: int = 220) -> dict:
    """Return a snippet around the best-matching query token plus highlight ranges."""
    tokens = tokenize(query)
    low = text.lower()
    pos = -1
    for tok in tokens:
        if len(tok) < 3:
            continue
        found = low.find(tok)
        if found != -1:
            pos = found
            break
    if pos == -1:
        snippet = text[: radius * 2]
        start = 0
    else:
        start = max(0, pos - radius)
        snippet = text[start: start + radius * 2]

    ranges = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        for m in re.finditer(re.escape(tok), snippet.lower()):
            ranges.append([m.start(), m.end()])
    ranges.sort()
    # merge overlapping ranges
    merged: list[list[int]] = []
    for r in ranges:
        if merged and r[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return {"snippet": snippet, "highlights": merged}

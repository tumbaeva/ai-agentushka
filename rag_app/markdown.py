"""Minimal, dependency-free Markdown -> HTML renderer.

Supports the subset used by the project docs: headings, paragraphs, fenced
code blocks, blockquotes, horizontal rules, ordered/unordered lists and tables,
plus inline code/bold/links. Intended for trusted local content (HELP.md).
"""

from __future__ import annotations

import html
import re

_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}.*$")
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_UL_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_OL_RE = re.compile(r"^\s*\d+\.\s+(.*)$")


def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def _is_table_start(lines: list[str], i: int) -> bool:
    return (
        i + 1 < len(lines)
        and "|" in lines[i]
        and _SEP_RE.match(lines[i + 1]) is not None
    )


def _split_row(row: str) -> list[str]:
    row = row.strip().strip("|")
    return [c.strip() for c in row.split("|")]


def render_md(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # fenced code
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1  # skip closing fence
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # horizontal rule
        if re.match(r"^\s*---+\s*$", line):
            out.append("<hr>")
            i += 1
            continue

        # heading
        m = _HEAD_RE.match(line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        # table
        if _is_table_start(lines, i):
            header = _split_row(lines[i])
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            html_tbl = ["<table class=\"md-table\"><thead><tr>"]
            html_tbl += [f"<th>{_inline(c)}</th>" for c in header]
            html_tbl.append("</tr></thead><tbody>")
            for r in rows:
                html_tbl.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            html_tbl.append("</tbody></table>")
            out.append("".join(html_tbl))
            continue

        # blockquote
        if line.strip().startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(_inline(lines[i].strip()[1:].strip()))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        # unordered list
        if _UL_RE.match(line):
            items = []
            while i < n and _UL_RE.match(lines[i]):
                items.append(f"<li>{_inline(_UL_RE.match(lines[i]).group(1))}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue

        # ordered list
        if _OL_RE.match(line):
            items = []
            while i < n and _OL_RE.match(lines[i]):
                items.append(f"<li>{_inline(_OL_RE.match(lines[i]).group(1))}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue

        # blank
        if not line.strip():
            i += 1
            continue

        # paragraph (gather consecutive plain lines)
        buf = []
        while i < n and lines[i].strip() and not _HEAD_RE.match(lines[i]) \
                and not lines[i].strip().startswith((">", "```")) \
                and not _UL_RE.match(lines[i]) and not _OL_RE.match(lines[i]) \
                and not re.match(r"^\s*---+\s*$", lines[i]) \
                and not _is_table_start(lines, i):
            buf.append(_inline(lines[i].strip()))
            i += 1
        out.append("<p>" + "<br>".join(buf) + "</p>")

    return "\n".join(out)

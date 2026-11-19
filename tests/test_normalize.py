from rag_app.normalize import normalize_whitespace, tokenize, highlight_fragment, shorten


def test_normalize_whitespace_collapses():
    assert normalize_whitespace("a\u00a0\u00a0b\n\n\n\nc") == "a b\n\nc"


def test_tokenize_keeps_cyrillic():
    assert tokenize("Номинации конкурса 2026!") == ["номинации", "конкурса", "2026"]


def test_highlight_fragment_marks_query():
    res = highlight_fragment("Сроки приёма заявок до 24 ноября 2026 года.", "сроки")
    assert "срок" in res["snippet"].lower()
    assert res["highlights"]


def test_shorten():
    assert shorten("x" * 300).endswith("...")

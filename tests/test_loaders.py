from pathlib import Path

from rag_app.loaders import load_knowledge_base_md, detect_title, LoadedDoc


def test_detect_title_picks_frequent_phrase():
    docs = [
        LoadedDoc(name="a", text="конкурс «Мир науки глазами детей – 2026» проходит", doc_type="md"),
        LoadedDoc(name="b", text="«Мир науки глазами детей – 2026» и снова «АлтГПУ»", doc_type="md"),
    ]
    assert detect_title(docs) == "Мир науки глазами детей – 2026"


def test_detect_title_empty():
    assert detect_title([LoadedDoc(name="a", text="без кавычек", doc_type="md")]) == ""


def test_splits_by_h1(tmp_path: Path):
    md = tmp_path / "knowledge_base.md"
    md.write_text(
        "# Информационное письмо.docx\n## Номинации\nТекст письма.\n"
        "# Положение о конкурсе рисунков_АлтГПУ_2026.docx\n## Сроки\nТекст положения.\n",
        encoding="utf-8",
    )
    docs = load_knowledge_base_md(md)
    names = {d.name for d in docs}
    assert "Информационное письмо.docx" in names
    assert "Положение о конкурсе рисунков_АлтГПУ_2026.docx" in names
    # internal sections stay inside the document text
    assert any("Номинации" in d.text for d in docs)


def test_missing_file_returns_empty(tmp_path: Path):
    assert load_knowledge_base_md(tmp_path / "nope.md") == []

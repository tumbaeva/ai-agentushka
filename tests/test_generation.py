from rag_app.generation import doc_overview, build_prompt


def test_doc_overview_uses_earliest_chunk_per_doc():
    chunks = [
        {"chunk_id": 5, "file_name": "A.md", "chunk_index": 1, "text": "second"},
        {"chunk_id": 0, "file_name": "A.md", "chunk_index": 0, "text": "Название конкурса здесь"},
        {"chunk_id": 3, "file_name": "B.md", "chunk_index": 0, "text": "Doc B intro"},
    ]
    ov = doc_overview(chunks, per_doc_chars=50)
    assert "Название конкурса здесь" in ov
    assert "A.md" in ov and "B.md" in ov


def test_build_prompt_includes_overview_and_context():
    retrieved = [{"file_name": "A.md", "section": "s", "text": "ctx", "chunk_id": 1, "score": 0.5}]
    msgs = build_prompt("вопрос", retrieved, overview="СПРАВКА-XYZ")
    content = msgs[-1]["content"]
    assert "СПРАВКА О ДОКУМЕНТАХ" in content
    assert "СПРАВКА-XYZ" in content
    assert "НАЙДЕННЫЕ ФРАГМЕНТЫ" in content
    assert "вопрос" in content

from rag_app.chunking import chunk_document, chunk_documents
from rag_app.loaders import LoadedDoc


def _doc():
    text = (
        "# Положение о конкурсе\n"
        "## Номинации\n"
        "Конкурс проводится по трём номинациям.\n"
        "Каждая номинация соответствует возрастной группе.\n"
        "## Сроки\n"
        "Приём заявок с 10 октября по 24 ноября 2026 года.\n"
        "## Награждение\n"
        "Победители получают дипломы.\n"
    )
    return LoadedDoc(name="Положение.docx", text=text, doc_type="docx")


def test_sections_are_tracked():
    chunks = chunk_document(_doc(), chunk_size=500, chunk_overlap=50, min_chunk_size=10)
    sections = {c.section for c in chunks}
    assert "Номинации" in sections
    assert "Награждение" in sections


def test_metadata_present():
    chunks = chunk_document(_doc(), chunk_size=500, chunk_overlap=50, min_chunk_size=10)
    assert all(c.file_name == "Положение.docx" for c in chunks)
    assert all(c.doc_type == "docx" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_long_paragraph_is_split():
    long_text = "Предложение номер один. " * 200
    doc = LoadedDoc(name="big.md", text=long_text, doc_type="md")
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=30, min_chunk_size=10)
    assert len(chunks) > 1
    assert all(c.chars <= 400 for c in chunks)


def test_chunk_documents_ids_are_continuous():
    docs = [_doc(), _doc()]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50, min_chunk_size=10)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))

from rag_app.eval import evaluate_one, run_eval, summarize, METRIC_COLS
from rag_app.eval import _parse_json_array, _derive_keywords, suggest_questions
from rag_app.loaders import LoadedDoc


class FakeIndex:
    def __init__(self, chunks):
        self.chunks = chunks


class FakeRetriever:
    def __init__(self, chunks, results_by_mode=None):
        self.index = FakeIndex(chunks)
        self._results = results_by_mode or {}

    def retrieve(self, query, mode="hybrid", k=4):
        return self._results.get(mode, [])


def _chunks():
    return [
        {"chunk_id": 0, "file_name": "Информационное письмо.docx", "text": "Номинации конкурса: три."},
        {"chunk_id": 1, "file_name": "Информационное письмо.docx", "text": "Сроки приёма заявок."},
        {"chunk_id": 2, "file_name": "Положение о конкурсе рисунков_АлтГПУ_2026.docx", "text": "Требования к файлам: до 10 МБ."},
    ]


def test_hit_and_doc_acc():
    results = [{"chunk_id": 0, "file_name": "Информационное письмо.docx", "text": "Номинации конкурса: три.", "score": 0.9}]
    retriever = FakeRetriever(_chunks(), {"hybrid": results})
    item = {"query": "номинации", "relevant_docs": {"Информационное письмо.docx"}, "keywords": ["номинац"]}
    metrics = evaluate_one(retriever, item, mode="hybrid", k=4)
    assert metrics["hit@k"] == 1.0
    assert metrics["doc_acc@k"] == 1.0
    assert metrics["mrr@k"] == 1.0


def test_miss_gives_zero():
    retriever = FakeRetriever(_chunks(), {"hybrid": []})
    item = {"query": "сроки", "relevant_docs": {"Информационное письмо.docx"}, "keywords": ["срок"]}
    metrics = evaluate_one(retriever, item, mode="hybrid", k=4)
    assert metrics["hit@k"] == 0.0
    assert metrics["mrr@k"] == 0.0


def test_run_and_summarize_shape():
    results = [{"chunk_id": 0, "file_name": "Информационное письмо.docx", "text": "Номинации конкурса", "score": 0.9}]
    retriever = FakeRetriever(_chunks(), {m: results for m in ["vector", "bm25", "hybrid", "multi_query", "multi_query_rerank"]})
    rows = run_eval(retriever, k=4)
    assert len(rows) == 4 * 5
    summary = summarize(rows)
    assert set(summary).issubset({"vector", "bm25", "hybrid", "multi_query", "multi_query_rerank"})
    for metrics in summary.values():
        assert set(metrics) == set(METRIC_COLS)


def test_parse_json_array_with_fence():
    txt = '```json\n[{"query":"q","docs":["a"]}]\n```'
    data = _parse_json_array(txt)
    assert data and data[0]["query"] == "q"


def test_parse_json_array_with_prose():
    txt = 'Вот вопросы: [{"query":"q","docs":[]}] — готово'
    assert _parse_json_array(txt)[0]["query"] == "q"


def test_derive_keywords_matches_corpus():
    docs = [LoadedDoc(name="d", text="Номинации конкурса утверждены", doc_type="md")]
    kws = _derive_keywords("Какие номинации утверждены?", docs)
    assert kws
    assert any("номина" in k for k in kws)


def test_suggest_questions_parses_and_grounds():
    from types import SimpleNamespace

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            content = '[{"query":"Какие номинации?","docs":["d"]}]'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    docs = [LoadedDoc(name="d", text="Номинации конкурса", doc_type="md")]
    qs = suggest_questions(FakeClient(), "m", docs, count=3, temperature=0.7)
    assert qs and qs[0]["query"] == "Какие номинации?"
    assert qs[0]["relevant_docs"] == ["d"]
    assert qs[0]["keywords"]

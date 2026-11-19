from rag_app.sweep import _score_avg, RANK_METRICS


def test_score_avg_is_mean_of_four_metrics():
    m = {"hit@k": 1.0, "doc_acc@k": 1.0, "recall@k": 0.5, "mrr@k": 0.5,
         "precision@k": 0.1, "doc_precision@k": 0.2}
    assert _score_avg(m) == 0.75


def test_score_avg_empty():
    assert _score_avg({}) == 0.0


def test_rank_metrics_contains_avg_and_columns():
    assert "score_avg" in RANK_METRICS
    assert "mrr@k" in RANK_METRICS

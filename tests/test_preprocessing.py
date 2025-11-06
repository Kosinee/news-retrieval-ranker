from src.metrics.ir_metrics import recall_at_k, ndcg_at_k

def test_recall_and_ndcg_basic():
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}
    results = {"q1": ["d1", "d3"], "q2": ["d4", "d3"]}

    recall = recall_at_k(qrels, results, k=2)
    ndcg = ndcg_at_k(qrels, results, k=2)

    assert 0 <= recall <= 1
    assert 0 <= ndcg <= 1

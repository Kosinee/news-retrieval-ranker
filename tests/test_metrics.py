import numpy as np
import pandas as pd
from src.metrics.ir_metrics import recall_at_k, ndcg_at_k

def test_ir_metrics_basic():
    qrels = {
        "Q1": {"D1": 1, "D2": 1},   # 👉 dict, как в BEIR
        "Q2": {"D3": 1}
    }
    results = {
        "Q1": ["D1", "D3"],
        "Q2": ["D3"]
    }

    recall = recall_at_k(qrels, results, k=2)
    ndcg = ndcg_at_k(qrels, results, k=2)

    assert 0 < recall <= 1
    assert 0 <= ndcg <= 1


def test_evaluate_mapping(monkeypatch):
    from src.training import evaluate_retriever as er
    import numpy as np

    # подменим faiss поиск и encode, чтобы не тянуть реальные модели
    def fake_encode_texts(model_dir, texts, batch_size=256):
        return np.random.randn(len(texts), 8).astype("float32")

    def fake_faiss_search(index, query_vecs, topk=10):
        # вернем индексы [0,1,2...]
        I = np.tile(np.arange(topk), (len(query_vecs), 1))
        return I.tolist()

    monkeypatch.setattr(er, "encode_texts", fake_encode_texts)
    monkeypatch.setattr(er, "faiss_search", fake_faiss_search)

    corpus = {f"D{i}": {"title": f"title{i}", "text": f"text{i}"} for i in range(100)}
    queries = {"Q1": "query one", "Q2": "query two"}
    qrels = {"Q1": {"D0": 1}, "Q2": {"D5": 1}}

    metrics = er.evaluate("models/mock", corpus, queries, qrels, k_vals=(5,))

    assert isinstance(metrics, dict)
    assert "recall@5" in metrics
    assert "ndcg@5" in metrics

import pytest
from src.training.train_retriever import build_pairs_from_qrels

@pytest.fixture
def fiqa_data():
    """Мини-версия данных, имитирующая формат FiQA из BEIR."""
    corpus = {
        "D1": {"title": "Stock market update", "text": "The stock market is up today."},
        "D2": {"title": "ETF investment", "text": "Exchange traded funds are popular."},
        "D3": {"title": "Real estate", "text": "Housing prices are stable."},
    }
    queries = {
        "Q1": "How to invest in ETFs?",
        "Q2": "What is the stock market doing?"
    }
    qrels = {
        "Q1": {"D2": 1},
        "Q2": {"D1": 1}
    }
    return corpus, queries, qrels


def test_train_pipeline_smoke(tmp_path, fiqa_data):
    corpus, queries, qrels = fiqa_data
    pairs = build_pairs_from_qrels(queries, corpus, qrels)

    # Проверяем, что сформировались пары query–doc
    assert len(pairs) == 2
    assert all(hasattr(p, "texts") for p in pairs)
    assert all(len(p.texts) == 2 for p in pairs)

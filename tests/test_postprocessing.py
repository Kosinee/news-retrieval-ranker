import numpy as np
import pytest
import faiss
from src.training.evaluate_retriever import faiss_search

@pytest.fixture
def fake_index():
    dim = 8
    index = faiss.IndexFlatIP(dim)
    vecs = np.random.randn(5, dim).astype("float32")
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index

@pytest.fixture
def fake_queries():
    q = np.random.randn(2, 8).astype("float32")
    faiss.normalize_L2(q)
    return q

def test_faiss_search_returns_lists(fake_index, fake_queries):
    res = faiss_search(fake_index, fake_queries, topk=3)
    assert isinstance(res, list)
    assert all(isinstance(r, list) for r in res)
    assert all(isinstance(i, int) for r in res for i in r)

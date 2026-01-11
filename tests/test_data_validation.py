import pytest

from src.validation.data_validation import (
    validate_corpus,
    validate_queries,
    validate_qrels,
)


def test_validate_corpus_structure():
    corpus = {"D1": {"title": "Title", "text": ""}}
    validate_corpus(corpus)

    bad = {"D2": {"title": "", "text": ""}}
    with pytest.raises(ValueError):
        validate_corpus(bad)


def test_validate_queries_required_fields():
    queries = {"Q1": "What is ETF?"}
    validate_queries(queries)

    bad = {"Q2": "   "}
    with pytest.raises(ValueError):
        validate_queries(bad)


def test_validate_qrels_types_and_range():
    qrels = {"Q1": {"D1": 1}}
    validate_qrels(qrels)

    bad_range = {"Q2": {"D2": -1}}
    with pytest.raises(ValueError):
        validate_qrels(bad_range)

    bad_type = {"Q3": {"D3": 0.5}}
    with pytest.raises(TypeError):
        validate_qrels(bad_type)

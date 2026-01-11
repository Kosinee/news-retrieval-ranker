from __future__ import annotations

from typing import Dict


def validate_corpus(corpus: Dict[str, Dict[str, str]]) -> None:
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError("Corpus must be a non-empty dict.")
    for doc_id, doc in corpus.items():
        if not doc_id:
            raise ValueError("Corpus doc_id is required.")
        if not isinstance(doc, dict):
            raise TypeError("Corpus entry must be a dict.")
        title = doc.get("title", "")
        text = doc.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise TypeError("Corpus title/text must be strings.")
        if not (title.strip() or text.strip()):
            raise ValueError("Corpus entry must contain title or text.")


def validate_queries(queries: Dict[str, str]) -> None:
    if not isinstance(queries, dict) or not queries:
        raise ValueError("Queries must be a non-empty dict.")
    for qid, text in queries.items():
        if not qid:
            raise ValueError("Query id is required.")
        if not isinstance(text, str):
            raise TypeError("Query text must be a string.")
        if not text.strip():
            raise ValueError("Query text must be non-empty.")


def validate_qrels(qrels: Dict[str, Dict[str, int]]) -> None:
    if not isinstance(qrels, dict) or not qrels:
        raise ValueError("Qrels must be a non-empty dict.")
    for qid, rels in qrels.items():
        if not qid:
            raise ValueError("Qrel query id is required.")
        if not isinstance(rels, dict) or not rels:
            raise ValueError("Qrel entries must be a non-empty dict.")
        for doc_id, score in rels.items():
            if not doc_id:
                raise ValueError("Qrel doc_id is required.")
            if not isinstance(score, int):
                raise TypeError("Qrel score must be int.")
            if score < 0:
                raise ValueError("Qrel score must be >= 0.")

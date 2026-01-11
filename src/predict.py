from __future__ import annotations

import argparse
import csv
import json
import os
import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

def read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus(corpus_path: str) -> Tuple[List[str], List[str]]:
    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    for row in read_jsonl(corpus_path):
        doc_id = str(row.get("_id", "")).strip()
        if not doc_id:
            continue
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        joined = (title + " " + text).strip()
        if not joined:
            continue
        corpus_ids.append(doc_id)
        corpus_texts.append(joined)
    return corpus_ids, corpus_texts


def load_queries(queries_path: str) -> Tuple[List[str], List[str]]:
    query_ids: List[str] = []
    query_texts: List[str] = []
    for row in read_jsonl(queries_path):
        qid = str(row.get("_id", "")).strip()
        text = str(row.get("text", "") or "").strip()
        if not qid or not text:
            continue
        query_ids.append(qid)
        query_texts.append(text)
    return query_ids, query_texts


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embs.astype("float32")


def resolve_model_path(config_path: str, model_path: str | None) -> str:
    if model_path:
        return model_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_path = (cfg.get("model") or {}).get("path")
    if not cfg_path:
        raise ValueError("Model path is required (use --model_path or set model.path in config).")
    return cfg_path


def load_id_text_map(path: str, text_key: str) -> Dict[str, str]:
    id_to_text: Dict[str, str] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("_id", "")).strip()
        if not item_id:
            continue
        text = str(row.get(text_key, "") or "").strip()
        if not text:
            continue
        id_to_text[item_id] = text
    return id_to_text


def load_corpus_map(path: str) -> Dict[str, str]:
    id_to_text: Dict[str, str] = {}
    for row in read_jsonl(path):
        doc_id = str(row.get("_id", "")).strip()
        if not doc_id:
            continue
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        joined = (title + " " + text).strip()
        if not joined:
            continue
        id_to_text[doc_id] = joined
    return id_to_text


def load_qrels(path: str) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row.get("query-id", "")).strip()
            doc_id = str(row.get("corpus-id", "")).strip()
            score = int(float(row.get("score", "0")))
            if not qid or not doc_id:
                continue
            rows.append((qid, doc_id, score))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to folder with corpus.jsonl, queries.jsonl, qrels")
    parser.add_argument("--output_path", required=True, help="Path to output CSV file or output directory")
    parser.add_argument("--config", default="config/retriever.yaml", help="Path to config YAML")
    parser.add_argument("--model_path", default=None, help="Optional model path override")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding")
    parser.add_argument("--limit", type=int, default=100, help="Number of qrels rows to score")
    parser.add_argument("--qrels_split", default="test", help="Qrels split to use: train/dev/test")
    args = parser.parse_args()

    setup_logging()

    model_path = resolve_model_path(args.config, args.model_path)

    corpus_path = os.path.join(args.input_path, "corpus.jsonl")
    queries_path = os.path.join(args.input_path, "queries.jsonl")
    qrels_path = os.path.join(args.input_path, "qrels", f"{args.qrels_split}.tsv")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

    corpus_map = load_corpus_map(corpus_path)
    query_map = load_id_text_map(queries_path, "text")
    if not corpus_map or not query_map:
        raise ValueError("Input data is empty after parsing.")
    qrels = load_qrels(qrels_path)
    if not qrels:
        raise ValueError("Qrels file is empty after parsing.")

    selected = qrels[: max(args.limit, 0)]
    relevant_map: Dict[str, set[str]] = {}
    for qid, doc_id, _ in qrels:
        relevant_map.setdefault(qid, set()).add(doc_id)

    corpus_ids = list(corpus_map.keys())
    pair_rows: List[Tuple[str, str, int]] = []
    for qid, doc_id, rel_score in selected:
        if qid not in query_map or doc_id not in corpus_map:
            continue
        pair_rows.append((qid, doc_id, int(rel_score)))

        rel_set = relevant_map.get(qid, set())
        neg_id = ""
        for cid in corpus_ids:
            if cid not in rel_set:
                neg_id = cid
                break
        if neg_id:
            pair_rows.append((qid, neg_id, 0))
    if not pair_rows:
        raise ValueError("No valid pairs found to score.")

    model = SentenceTransformer(model_path)
    unique_qids = sorted({qid for qid, _, _ in pair_rows})
    unique_docids = sorted({doc_id for _, doc_id, _ in pair_rows})
    query_vecs = encode_texts(model, [query_map[qid] for qid in unique_qids], args.batch_size)
    doc_vecs = encode_texts(model, [corpus_map[doc_id] for doc_id in unique_docids], args.batch_size)
    qid_to_idx = {qid: i for i, qid in enumerate(unique_qids)}
    docid_to_idx = {doc_id: i for i, doc_id in enumerate(unique_docids)}

    output_path = args.output_path
    output_dir = None
    if output_path.endswith(os.sep):
        output_dir = output_path
    elif os.path.isdir(output_path):
        output_dir = output_path
    if output_dir is not None:
        output_path = os.path.join(output_dir, "preds.csv")
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "preds.csv")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "doc_id", "label", "score"])
        for qid, doc_id, label in pair_rows:
            q_idx = qid_to_idx[qid]
            d_idx = docid_to_idx[doc_id]
            score = float(np.dot(query_vecs[q_idx], doc_vecs[d_idx]))
            writer.writerow([qid, doc_id, label, score])
    logger.info("[predict] Wrote predictions to %s", output_path)


if __name__ == "__main__":
    main()

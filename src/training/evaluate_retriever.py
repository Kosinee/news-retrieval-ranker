from __future__ import annotations
import json
from datetime import datetime
import time
import yaml
import os
import logging
import numpy as np
from typing import Dict, List, Optional
import faiss
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

from src.metrics.ir_metrics import compute_metrics
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def encode_texts(model_dir: str, texts: List[str], batch_size: int = 256) -> np.ndarray:
    model = SentenceTransformer(model_dir)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embs.astype("float32")


def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embs.shape[1])
    faiss.normalize_L2(embs)
    index.add(embs)
    return index


def faiss_search(index: faiss.IndexFlatIP, query_vecs: np.ndarray, topk: int = 100) -> List[List[int]]:
    faiss.normalize_L2(query_vecs)
    D, I = index.search(query_vecs, topk)
    return I.tolist()


def evaluate(
    model_path: str,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    k_vals: tuple[int] = (10, 100),
    corpus_limit: Optional[int] = None,
    query_limit: Optional[int] = None,
    faiss_threads: Optional[int] = None,
) -> Dict[str, float]:
    if faiss_threads is not None:
        try:
            faiss.omp_set_num_threads(faiss_threads)
        except Exception:
            pass
    t0 = time.time()
    corpus_ids = list(corpus.keys())
    if corpus_limit is not None and corpus_limit > 0:
        corpus_ids = corpus_ids[:corpus_limit]
    corpus_texts = [
        (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
        for cid in corpus_ids
    ]
    corpus_vecs = encode_texts(model_path, corpus_texts)
    index = build_faiss_index(corpus_vecs)
    id_map = {i: corpus_ids[i] for i in range(len(corpus_ids))}
    query_ids = list(queries.keys())
    if query_limit is not None and query_limit > 0:
        query_ids = query_ids[:query_limit]
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    query_texts = [queries[qid] for qid in query_ids]
    query_vecs = encode_texts(model_path, query_texts)
    I = faiss_search(index, query_vecs, topk=max(k_vals))
    results = {qid: [id_map[i] for i in row if i in id_map] for qid, row in zip(query_ids, I)}
    metrics = compute_metrics(qrels, results, k_vals)
    logger.info("[eval] Total evaluate: %.2fs", time.time() - t0)

    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/retriever.yaml", help="Path to retriever.yaml")
    ap.add_argument("--output", default=None, help="Optional metrics output path")
    args = ap.parse_args()

    setup_logging()

    cfg = yaml.safe_load(open(args.config))
    model_path = cfg["model"]["path"]
    data_path = cfg["dataset"]["path"]
    if args.output:
        metrics_path = args.output
    else:
        metrics_path = cfg["train"]["metrics_path"] + "/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    metrics = evaluate(model_path, corpus, queries, qrels, k_vals=(10, 100))

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    logger.info("Metrics: %s", metrics)

from __future__ import annotations
import json
from datetime import datetime
import yaml
import numpy as np
from typing import Dict, List
import faiss
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

from src.metrics.ir_metrics import compute_metrics


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
) -> Dict[str, float]:
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
        for cid in corpus_ids
    ]
    doc_vecs = encode_texts(model_path, corpus_texts)
    index = build_faiss_index(doc_vecs)
    id_map = {i: corpus_ids[i] for i in range(len(corpus_ids))}
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_vecs = encode_texts(model_path, query_texts)
    I = faiss_search(index, query_vecs, topk=max(k_vals))
    results = {qid: [id_map[i] for i in row if i in id_map] for qid, row in zip(query_ids, I)}
    metrics = compute_metrics(qrels, results, k_vals)

    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/retriever.yaml", help="Path to retriever.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_path = cfg["model"]["path"]
    data_path = cfg["dataset"]["path"]
    metrics_path = cfg["train"]["metrics_path"] + "/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    metrics = evaluate(model_path, corpus, queries, qrels, k_vals=[10, 100])

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(metrics)

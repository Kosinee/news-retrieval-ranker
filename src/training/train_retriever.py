from __future__ import annotations
import os
import random
import json
import time
from datetime import datetime
import yaml
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Sampler
from beir.datasets.data_loader import GenericDataLoader
import torch
import mlflow
import mlflow.transformers


def build_pairs_from_qrels(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
) -> tuple[List[InputExample], Dict[str, List[int]]]:
    pairs = []
    qid_to_idxs: Dict[str, List[int]] = {}

    for qid, rel_docs in qrels.items():
        query_text = queries.get(qid)
        if not query_text:
            continue

        for docid, rel in rel_docs.items():
            if rel > 0 and docid in corpus:
                doc_entry = corpus[docid]
                doc_text = (doc_entry.get("title", "") + " " + doc_entry.get("text", "")).strip()
                if doc_text:
                    pairs.append(InputExample(texts=[query_text, doc_text]))
                    qid_to_idxs.setdefault(qid, []).append(len(pairs) - 1)
    return pairs, qid_to_idxs


class UniqueQuerySampler(Sampler[int]):
    def __init__(self, qid_to_idxs: Dict[str, List[int]]):
        self.qid_to_idxs = qid_to_idxs

    def __iter__(self):
        qids = list(self.qid_to_idxs.keys())
        random.shuffle(qids)
        for qid in qids:
            idx = random.choice(self.qid_to_idxs[qid])
            yield idx

    def __len__(self) -> int:
        return len(self.qid_to_idxs)


def load_dvc_hashes(dvc_lock_path: str, data_path: str, model_path: str) -> Dict[str, str]:
    if not os.path.exists(dvc_lock_path):
        return {}

    with open(dvc_lock_path, "r") as f:
        dvc_lock = yaml.safe_load(f) or {}

    def find_md5(target_path: str) -> Optional[str]:
        for stage in (dvc_lock.get("stages") or {}).values():
            for section in ("deps", "outs"):
                for entry in stage.get(section, []):
                    if entry.get("path") == target_path and entry.get("md5"):
                        return entry["md5"]
        return None

    hashes = {}
    data_hash = find_md5(data_path)
    model_hash = find_md5(model_path)
    if data_hash:
        hashes["dvc_data_hash"] = data_hash
    if model_hash:
        hashes["dvc_model_hash"] = model_hash
    return hashes


def log_artifact_if_exists(path: str, artifact_path: Optional[str] = None) -> None:
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        mlflow.log_artifacts(path, artifact_path=artifact_path)
    else:
        mlflow.log_artifact(path, artifact_path=artifact_path)


def train_bi_encoder(
    model_name: str,
    pairs: List[InputExample],
    qid_to_idxs: Dict[str, List[int]],
    model_path: str,
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 2e-5,
    max_seq_length: int = 128,
):
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length

    sampler = UniqueQuerySampler(qid_to_idxs)
    loader = DataLoader(pairs, sampler=sampler, batch_size=batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    warmup = int(0.1 * max(1, len(loader)))

    use_amp = True
    if torch.backends.mps.is_available() and torch.__version__ < "2.5.0":
        use_amp = False

    model.fit(
        [(loader, loss_fn)],
        epochs=epochs,
        optimizer_params={"lr": lr},
        warmup_steps=warmup,
        use_amp=use_amp,
        output_path=model_path,
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/retriever.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["dataset"]["path"]
    model_name = cfg["model"]["name"]
    model_path = cfg["model"]["path"]

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")

    pairs, qid_to_idxs = build_pairs_from_qrels(queries, corpus, qrels)

    mlflow.set_experiment("news-retrieval-ranker")
    mlflow.transformers.autolog()

    with mlflow.start_run() as run:
        eval_corpus_limit = int(os.getenv("EVAL_CORPUS_LIMIT", "10000"))
        eval_query_limit = int(os.getenv("EVAL_QUERY_LIMIT", "200"))
        eval_topk = int(os.getenv("EVAL_TOPK", "10"))
        eval_faiss_threads = int(os.getenv("EVAL_FAISS_THREADS", "1"))
        mlflow.log_params(
            {
                "model_name": model_name,
                "model_path": model_path,
                "data_path": data_path,
                "batch_size": int(cfg["train"]["batch_size"]),
                "epochs": int(cfg["train"]["epochs"]),
                "lr": float(cfg["train"]["lr"]),
                "max_seq_length": int(cfg["train"]["max_seq_length"]),
                "pairs_count": len(pairs),
                "unique_queries": len(qid_to_idxs),
                "config_path": args.config,
                "eval_corpus_limit": eval_corpus_limit,
                "eval_query_limit": eval_query_limit,
                "eval_topk": eval_topk,
                "eval_faiss_threads": eval_faiss_threads,
            }
        )

        dvc_hashes = load_dvc_hashes("dvc.lock", data_path, model_path)
        for tag_name, tag_value in dvc_hashes.items():
            mlflow.set_tag(tag_name, tag_value)

        t0 = time.time()
        train_bi_encoder(
            model_name=model_name,
            pairs=pairs,
            qid_to_idxs=qid_to_idxs,
            model_path=model_path,
            batch_size=int(cfg["train"]["batch_size"]),
            epochs=int(cfg["train"]["epochs"]),
            lr=float(cfg["train"]["lr"]),
            max_seq_length=int(cfg["train"]["max_seq_length"]),
        )
        print(f"[train] Training: {time.time() - t0:.2f}s")

        from src.training.evaluate_retriever import evaluate

        t1 = time.time()
        corpus_test, queries_test, qrels_test = GenericDataLoader(data_folder=data_path).load(split="test")
        print(f"[train] Load test split: {time.time() - t1:.2f}s")
        t2 = time.time()
        metrics = evaluate(
            model_path,
            corpus_test,
            queries_test,
            qrels_test,
            k_vals=(eval_topk,),
            corpus_limit=eval_corpus_limit,
            query_limit=eval_query_limit,
            faiss_threads=eval_faiss_threads,
        )
        print(f"[train] Evaluate: {time.time() - t2:.2f}s")
        safe_metrics = {k.replace("@", "_at_"): v for k, v in metrics.items()}
        mlflow.log_metrics(safe_metrics)

        metrics_dir = cfg["train"]["metrics_path"]
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(
            metrics_dir, f"metrics_{run.info.run_id}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        t3 = time.time()
        log_artifact_if_exists(metrics_path, artifact_path="metrics")
        log_artifact_if_exists(model_path, artifact_path="model")
        log_artifact_if_exists("dvc.lock")
        log_artifact_if_exists(args.config, artifact_path="config")
        print(f"[train] Log artifacts: {time.time() - t3:.2f}s")

from __future__ import annotations
import os
import yaml
from typing import List, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from beir.datasets.data_loader import GenericDataLoader


def build_pairs_from_qrels(
    queries: Dict[str, str],
    corpus: Dict[str, Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
) -> List[InputExample]:
    pairs = []

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
    return pairs


def train_bi_encoder(
    model_name: str,
    pairs: List[InputExample],
    model_path: str,
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 2e-5,
    max_seq_length: int = 128,
):
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length

    loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    warmup = int(0.1 * max(1, len(loader)))

    model.fit(
        [(loader, loss_fn)],
        epochs=epochs,
        optimizer_params={"lr": lr},
        warmup_steps=warmup,
        use_amp=True,
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

    pairs = build_pairs_from_qrels(queries, corpus, qrels)

    train_bi_encoder(
        model_name=model_name,
        pairs=pairs,
        model_path=model_path,
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        max_seq_length=cfg["train"]["max_seq_length"],
    )

from __future__ import annotations
import os
import yaml
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import logging


def load_beir_dataset(dataset_name, data_path):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()]
    )
    if not os.path.exists(os.path.join(data_path, "corpus.jsonl")):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        util.download_and_unzip(url, data_path)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    return corpus, queries, qrels


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/retriever.yaml", help="Path to retriever.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_path = cfg["dataset"]["path"]
    dataset_name = cfg["dataset"]["name"]
    corpus, queries, qrels = load_beir_dataset(dataset_name, data_path)

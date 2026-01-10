from __future__ import annotations

import argparse
import os
import shutil
import yaml


def copy_tree(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for entry in os.listdir(src_dir):
        src_path = os.path.join(src_dir, entry)
        dst_path = os.path.join(dst_dir, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def prepare_dataset(raw_path: str, prepared_path: str) -> None:
    os.makedirs(prepared_path, exist_ok=True)
    copy_tree(raw_path, prepared_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/retriever.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg["dataset"]["raw_path"]
    prepared_path = cfg["dataset"]["path"]

    prepare_dataset(raw_path, prepared_path)

import argparse
import os

import torch
from transformers import AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="static/artifacts/retriever_fiqa",
        help="Path to a HuggingFace-compatible model directory",
    )
    parser.add_argument(
        "--output",
        default="torchserve/model_files/model.pt",
        help="Where to save the model state_dict",
    )
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_dir)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[export] Saved state_dict to {args.output}")


if __name__ == "__main__":
    main()

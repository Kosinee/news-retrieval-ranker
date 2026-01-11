#!/usr/bin/env bash
set -euo pipefail

MODEL_SRC="${1:-static/artifacts/retriever_fiqa}"
STAGE_DIR="torchserve/model_files"
MODEL_STORE="model-store"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}" "${MODEL_STORE}"
cp -R "${MODEL_SRC}/." "${STAGE_DIR}/"

python scripts/export_torchserve.py --model_dir "${MODEL_SRC}" --output "${STAGE_DIR}/model.pt"

torch-model-archiver \
  --model-name mymodel \
  --version 1.0 \
  --serialized-file "${STAGE_DIR}/model.pt" \
  --handler torchserve/handler.py \
  --extra-files "${STAGE_DIR}" \
  --export-path "${MODEL_STORE}" \
  --force

echo "[torchserve] Created ${MODEL_STORE}/mymodel.mar"

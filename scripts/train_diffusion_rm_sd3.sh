#!/usr/bin/env sh
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Required environment variables:
#   SD3_MODEL_PATH                Base SD3 model path
#   DIFFUSION_RM_CHECKPOINT_PATH  Diffusion-RM checkpoint directory
#   DIFFUSION_RM_CONFIG_PATH      Diffusion-RM config path
#
# Optional:
#   WORKER_NUM (default: 8)
#   MASTER_PORT (default: 9527)
#   DATASET_DIR (default: ${ROOT_DIR}/dataset/ocr)
#   OUTPUT_NAME (default: diffusion-rm-sd3)

: "${SD3_MODEL_PATH:?Please set SD3_MODEL_PATH}"
: "${DIFFUSION_RM_CHECKPOINT_PATH:?Please set DIFFUSION_RM_CHECKPOINT_PATH}"
: "${DIFFUSION_RM_CONFIG_PATH:?Please set DIFFUSION_RM_CONFIG_PATH}"

WORKER_NUM="${WORKER_NUM:-8}"
MASTER_PORT="${MASTER_PORT:-9527}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/dataset/ocr}"
OUTPUT_NAME="${OUTPUT_NAME:-diffusion-rm-sd3}"

export TOKENIZERS_PARALLELISM=False

cd "${ROOT_DIR}"

msrun --worker_num "${WORKER_NUM}" --local_worker_num "${WORKER_NUM}" --master_port "${MASTER_PORT}" --join True \
  "${ROOT_DIR}/scripts/train_sd3.py" \
  --run-name "${OUTPUT_NAME}" \
  --reward diffusion-rm-sd3 \
  --reward-weights 1.0 \
  --model "${SD3_MODEL_PATH}" \
  --dataset "${DATASET_DIR}" \
  --guidance-scale 1.0 \
  --train-batch-size 1 \
  --num-image-per-prompt 1 \
  --test-batch-size 1 \
  --beta 0.0 \
  --diffusion-rm-checkpoint-path "${DIFFUSION_RM_CHECKPOINT_PATH}" \
  --diffusion-rm-config-path "${DIFFUSION_RM_CONFIG_PATH}" \
  --diffusion-rm-u 0.9

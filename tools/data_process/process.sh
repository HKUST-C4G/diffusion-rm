#!/bin/bash

# 你的数据集和输出路径
DATASET_PATH="/path/to/your/hpsv3_dataset"
OUT_DIR="/path/to/your/output_latents"
META_DIR="/path/to/your/images/base/dir"

echo "Starting single-node multi-GPU extraction..."

torchrun --nproc_per_node=8 \
    prepare_dataset.py \
    --dataset_path="${DATASET_PATH}" \
    --out_dir="${OUT_DIR}" \
    --meta_dir="${META_DIR}" \
    --dataset_name="hpsv3" \
    --split="train" \
    --batch_size=1 \
    --num_workers=8
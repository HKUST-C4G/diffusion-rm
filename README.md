<div align="center">

# Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2602.11146-b31b1b.svg)](https://arxiv.org/abs/2602.11146)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/HKUST-C4G/diffusion-rm)
[![MindSpore](https://img.shields.io/badge/MindSpore-Ascend-red)](https://github.com/mindspore-ai/mindspore)

</div>

This branch provides the [MindSpore](https://github.com/mindspore-ai/mindspore) / Ascend implementation for the SD3 alignment stage. It keeps the Diffusion-RM reward semantics aligned with the original MindSpore checkpoint and uses Diffusion-RM as the reward model for Flow-GRPO training.

> A diffusion-native latent reward model for efficient diffusion model alignment.

## Open-Source Roadmap

- [x] MindSpore / Ascend source code for SD3 Flow-GRPO alignment with Diffusion-RM reward.
- [x] SD3 Diffusion-RM checkpoint support in MindSpore `.ckpt` format.
- [x] Prompt dataset and training entrypoint for SD3 alignment.
- [ ] Additional MindSpore backbones such as FLUX and Z-Image.
- [ ] Evaluation code and logistic normalization.

## Installation

Create a MindSpore environment that matches your Ascend/CANN installation, then install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The core dependencies are `mindspore`, `mindone`, `numpy`, `pillow`, `tqdm`, and `pyyaml`.

## Checkpoint

The SD3 Diffusion-RM reward head checkpoint is available here:

[Download `rm_head.ckpt`](https://drive.google.com/file/d/1jNuhGNksemVJO59NhU0NFCdDRKhzEdu2/view?usp=sharing)

Place the downloaded file under a checkpoint directory named `rm_head.ckpt`, for example:

```text
checkpoints/diffusion_rm_sd3/
└── rm_head.ckpt
```

Use this directory as `DIFFUSION_RM_CHECKPOINT_PATH` when launching training.

## Data

The data setup follows the main branch. For reward model training data, the preprocessed SD3.5-Medium dataset is available at:

[DiNa-LRM-SD35m-HPSv3-Preprocess-Data](https://huggingface.co/datasets/liuhuohuo/DiNa-LRM-SD35m-HPSv3-Preprocess-Data)

For the alignment stage in this branch, the included prompt dataset is under:

```text
dataset/ocr/
├── train.txt
└── test.txt
```

## SD3 Alignment Training

Set the required paths:

```bash
export SD3_MODEL_PATH=/path/to/stable-diffusion-3.5-medium
export DIFFUSION_RM_CHECKPOINT_PATH=/path/to/checkpoints/diffusion_rm_sd3
export DIFFUSION_RM_CONFIG_PATH=/path/to/diffusion-rm/config.json
```

Then launch training:

```bash
chmod +x scripts/train_diffusion_rm_sd3.sh
scripts/train_diffusion_rm_sd3.sh
```

Optional environment variables:

- `WORKER_NUM`: number of Ascend workers, default `8`.
- `MASTER_PORT`: `msrun` master port, default `9527`.
- `DATASET_DIR`: prompt dataset path, default `dataset/ocr`.
- `OUTPUT_NAME`: output run name, default `diffusion-rm-sd3`.

The launch script runs:

```bash
msrun --worker_num "${WORKER_NUM}" --local_worker_num "${WORKER_NUM}" \
  --master_port "${MASTER_PORT}" --join True \
  scripts/train_sd3.py \
  --reward diffusion-rm-sd3 \
  --reward-weights 1.0 \
  --model "${SD3_MODEL_PATH}" \
  --dataset "${DATASET_DIR}" \
  --diffusion-rm-checkpoint-path "${DIFFUSION_RM_CHECKPOINT_PATH}" \
  --diffusion-rm-config-path "${DIFFUSION_RM_CONFIG_PATH}" \
  --diffusion-rm-u 0.9
```

## Inference

After training, generate images from prompts with the base SD3 model and an optional trained LoRA:

```bash
python scripts/infer_sd3.py \
  --model-id /path/to/stable-diffusion-3.5-medium \
  --lora-path /path/to/output/checkpoints/step_xxx/backbone_lora \
  --prompt-file dataset/ocr/test.txt \
  --output-dir outputs/infer \
  --num-inference-steps 40 \
  --guidance-scale 4.5 \
  --dtype fp16
```

If you do not use a LoRA checkpoint, omit `--lora-path`.

## Repository Layout

```text
.
├── flow_grpo/
│   ├── trainer/                     # MindSpore Flow-GRPO training components
│   ├── scorer/                      # Diffusion-RM scorer and aggregation
│   ├── optim/                       # BF16 AdamW optimizer
│   ├── dataset.py                   # Prompt dataset and distributed sampler
│   └── utils.py
├── scripts/
│   ├── train_sd3.py                 # SD3 Flow-GRPO training entrypoint
│   ├── train_diffusion_rm_sd3.sh    # Recommended launch script
│   └── infer_sd3.py                 # SD3 inference script
├── config/
│   └── diffusion_rm_example.yaml
├── dataset/
│   └── ocr/
└── requirements.txt
```

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{liu2026beyond,
  title={Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling},
  author={Liu, Gongye and Yang, Bo and Zhi, Yida and Zhong, Zhizhou and Ke, Lei and Deng, Didan and Gao, Han and Huang, Yongxiang and Zhang, Kaihao and Fu, Hongbo and others},
  journal={arXiv preprint arXiv:2602.11146},
  year={2026}
}
```

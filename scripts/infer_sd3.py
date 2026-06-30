import argparse
import os

import mindspore as ms
import numpy as np
from mindone.diffusers import StableDiffusion3Pipeline
from mindone.peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="SD3 inference script for Flow-GRPO-MS outputs."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base SD3 model path or model id.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional LoRA checkpoint directory (e.g. .../backbone_lora).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Text file with one prompt per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/infer",
        help="Directory to save generated images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Inference dtype for SD3 pipeline.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "fp16":
        return ms.float16
    if dtype_name == "bf16":
        return ms.bfloat16
    return ms.float32


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, ms_dtype=dtype)
    if args.lora_path:
        if not os.path.isdir(args.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
        pipe.transformer.merge_and_unload()

    for idx, prompt in enumerate(prompts):
        generator = np.random.default_rng(args.seed + idx)
        output = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
            return_dict=True,
        )
        output.images[0].save(os.path.join(args.output_dir, f"{idx:06d}.png"))

    print(f"Saved {len(prompts)} images to: {args.output_dir}")


if __name__ == "__main__":
    main()

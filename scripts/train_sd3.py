import argparse
import datetime
import os
import sys
import copy
from collections import defaultdict
from typing import Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn
import numpy as np
from mindone.diffusers import FlowMatchEulerDiscreteScheduler
from mindone.peft import LoraConfig, PeftModel, get_peft_model
from mindspore.dataset import DictIterator, GeneratorDataset
from tqdm import tqdm, trange

from flow_grpo.dataset import DistributedKRepeatSampler, TextPromptDataset
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.logging import get_logger
from flow_grpo.optim import BF16AdamW
from flow_grpo.scorer import AVAILABLE_SCORERS, MultiScorer
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.trainer import (
    FlowMatchEulerSDEDiscreteScheduler,
    NetWithLoss,
    StableDiffusion3PipelineWithSDELogProb,
)
from flow_grpo.utils import (
    clip_by_global_norm,
    gather,
    map_,
    requires_grad_,
    save_checkpoint,
    syn_gradients,
)

DEFAULT_MODEL = "stabilityai/stable-diffusion-3.5-medium"

logger = get_logger()

def evaluate(
    pipeline: StableDiffusion3PipelineWithSDELogProb,
    reward_fn: MultiScorer,
    test_iter: DictIterator,
    sample_neg_prompt_embeds: ms.Tensor,
    sample_neg_pooled_prompt_embeds: ms.Tensor,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
    outdir: str = "output",
    ema: Optional[EMAModuleWrapper] = None,
    total_num: Optional[int] = None,
    eval_num_steps: int = 40,
    guidance_scale: float = 1.0,
    resolution: int = 512,
    max_sequence_length: int = 128,
    seed: int = 0,
) -> None:
    if ema:
        # copy the ema parameters to the model
        ema.copy_ema_to_model()

    if scheduler is not None:
        # change the SDE scheduler back to FlowMatchEulerDiscreteScheduler
        sde_scheduler = pipeline.scheduler
        pipeline.scheduler = scheduler

    # set the pipeline to evaluation mode
    pipeline.transformer.set_train(False)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    total_prompts = list()
    all_rewards = defaultdict(list)

    # generate the images with the same initials noise for each evaluation step
    generator = np.random.default_rng(seed)
    for i, test_batch in tqdm(
        enumerate(test_iter), desc="Eval: ", total=total_num, dynamic_ncols=True
    ):
        prompts = test_batch["prompt"].tolist()
        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompts,
            None,
            None,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )

        output = pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
            num_inference_steps=eval_num_steps,
            guidance_scale=guidance_scale,
            output_type="pil",
            return_dict=True,
            height=resolution,
            width=resolution,
            generator=generator,
        )

        # save the image for visualization
        for j, (prompt, image) in enumerate(zip(prompts, output.images)):
            num = i * len(prompts) + j
            fname = f"{num}.jpg"
            total_prompts.append((fname, prompt))
            image.save(os.path.join(outdir, fname))

        # calculate the validation reward
        rewards = reward_fn(output.images, prompts)
        for k, v in rewards.items():
            all_rewards[k].extend(v)

    avg_rewards = dict()
    for k, v in all_rewards.items():
        avg_rewards[k] = np.mean(v).item()

    logger.info(f"Validation rewards: {avg_rewards}")
    with open(os.path.join(outdir, "prompt.txt"), "w") as f:
        for fname, prompt in total_prompts:
            f.write(f"{fname},{prompt}\n")

    if ema:
        ema.copy_temp_to_model()

    if scheduler is not None:
        pipeline.scheduler = sde_scheduler


def train(args: argparse.Namespace):
    ms.set_seed(args.seed)

    dist.init_process_group()
    num_processes = dist.get_world_size()
    process_index = dist.get_rank()
    is_main_process = process_index == 0

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id
    output_dir = os.path.join("output", args.run_name)

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(args.num_steps * args.timestep_fraction)

    logger.info(f"\n{args}")

    with nn.no_init_parameters():
        pipeline = StableDiffusion3PipelineWithSDELogProb.from_pretrained(args.model)

    original_scheduler = pipeline.scheduler
    scheduler_config = FlowMatchEulerSDEDiscreteScheduler.load_config(
        args.model, subfolder="scheduler"
    )
    pipeline.scheduler = FlowMatchEulerSDEDiscreteScheduler.from_config(
        scheduler_config
    )

    requires_grad_(pipeline.vae, False)
    requires_grad_(pipeline.text_encoder, False)
    requires_grad_(pipeline.text_encoder_2, False)
    requires_grad_(pipeline.text_encoder_3, False)
    requires_grad_(pipeline.transformer, not args.use_lora)

    pipeline.safety_checker = None

    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = ms.float32
    if args.mixed_precision == "fp16":
        inference_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        inference_dtype = ms.bfloat16

    pipeline.vae.to(ms.float32)
    pipeline.text_encoder.to(inference_dtype)
    pipeline.text_encoder_2.to(inference_dtype)
    pipeline.text_encoder_3.to(inference_dtype)

    if args.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if args.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, args.lora_path
            )
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(
                pipeline.transformer, transformer_lora_config
            )
    pipeline.transformer.to(inference_dtype)

    trainable_parameters = ms.ParameterTuple(
        filter(lambda p: p.requires_grad, pipeline.transformer.get_parameters())
    )

    transformer_params = sum(
        [param.size for param in pipeline.transformer.get_parameters()]
    )
    vae_params = sum([param.size for param in pipeline.vae.get_parameters()])
    text_encoder_params = sum(
        [param.size for param in pipeline.text_encoder.get_parameters()]
    )
    text_encoder_2_params = sum(
        [param.size for param in pipeline.text_encoder_2.get_parameters()]
    )
    text_encoder_3_params = sum(
        [param.size for param in pipeline.text_encoder_3.get_parameters()]
    )
    total_params = (
        transformer_params
        + vae_params
        + text_encoder_params
        + text_encoder_2_params
        + text_encoder_3_params
    )
    trainable_params = sum([param.size for param in trainable_parameters])

    logger.info(
        f"Total num. of parameters: {total_params:,} (transformer: {transformer_params:,}, vae: {vae_params:,}, tex_encoder: {text_encoder_params:,}, text_encoder_2: {text_encoder_2_params:,}, text_encoder_3: {text_encoder_3_params:,})"
    )
    logger.info(f"Total num. of trainable parameters: {trainable_params:,}")

    ema = (
        EMAModuleWrapper(trainable_parameters, decay=0.9, update_step_interval=1)
        if args.ema
        else None
    )

    optimizer = BF16AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # prepare prompt and reward fn
    if args.reward_weights is not None:
        if len(args.reward_weights) != len(args.reward):
            raise ValueError(
                f"--reward-weights length ({len(args.reward_weights)}) must match "
                f"--reward length ({len(args.reward)})."
            )
        weights = list(map(float, args.reward_weights))
    else:
        weights = [1 / len(args.reward)] * len(args.reward)
    scorers = dict(zip(args.reward, weights))
    logger.info("Using scorers: %s", scorers)
    
    # prepare scorer configs for Diffusion-RM
    scorer_configs = {}
    for reward_name in args.reward:
        if reward_name in ["diffusion-rm-flux", "diffusion-rm-sd3"]:
            if not args.diffusion_rm_checkpoint_path or not args.diffusion_rm_config_path:
                raise ValueError(
                    f"diffusion_rm_checkpoint_path and diffusion_rm_config_path must be provided when using {reward_name}"
                )
            cfg = {
                "checkpoint_path": args.diffusion_rm_checkpoint_path,
                "config_path": args.diffusion_rm_config_path,
                "u": args.diffusion_rm_u,
            }
            # 推荐 SD3 场景复用训练时 pipeline（节省显存/内存）；同时用 shallow copy 避免改动训练 scheduler
            if reward_name == "diffusion-rm-sd3":
                rm_pipeline = copy.copy(pipeline)
                rm_pipeline.scheduler = original_scheduler
                cfg["pipeline"] = rm_pipeline
                # pipeline 已提供则无需 pipeline_path
            else:
                cfg["pipeline_path"] = args.diffusion_rm_pipeline_path or args.model
            scorer_configs[reward_name] = cfg
    
    reward_fn = MultiScorer(scorers, scorer_configs)

    train_dataset = TextPromptDataset(args.dataset, "train")
    test_dataset = TextPromptDataset(args.dataset, "test", max_num=args.validation_num)
    train_sampler = DistributedKRepeatSampler(
        batch_size=args.train_batch_size,
        k=args.num_image_per_prompt,
        num_shards=num_processes,
        shard_id=process_index,
        num_iters=args.num_epochs * args.num_batches_per_epoch,
    )

    train_dataloader = GeneratorDataset(
        train_dataset,
        column_names="prompt",
        batch_sampler=train_sampler,
        num_parallel_workers=1,
    )

    test_dataloader = GeneratorDataset(
        test_dataset, column_names="prompt", num_parallel_workers=1, shuffle=False
    )
    test_dataloader = test_dataloader.batch(
        args.test_batch_size, num_parallel_workers=1, drop_remainder=False
    )

    neg_prompt_embed, _, neg_pooled_prompt_embed, _ = pipeline.encode_prompt(
        "",
        None,
        None,
        do_classifier_free_guidance=False,
        max_sequence_length=args.max_sequence_length,
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        args.train_batch_size, 1
    )
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        args.train_batch_size, 1
    )

    if args.num_image_per_prompt == 1 and args.per_prompt_stat_tracking:
        args.per_prompt_stat_tracking = False

    if args.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(args.global_std)

    samples_per_epoch = (
        args.train_batch_size * num_processes * args.num_batches_per_epoch
    )
    total_train_batch_size = (
        args.train_batch_size * num_processes * args.gradient_accumulation_steps
    )

    logger.info(f"Num Epochs = {args.num_epochs}")
    logger.info(f"Train batch size  = {args.train_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"Number of inner epochs = {args.num_inner_epochs}")

    first_epoch = 0
    global_step = 0

    train_iter = train_dataloader.create_dict_iterator(output_numpy=True)
    test_iter = test_dataloader.create_dict_iterator(output_numpy=True)

    net_with_loss = NetWithLoss(
        pipeline,
        guidance_scale=args.guidance_scale,
        adv_clip_max=args.adv_clip_max,
        beta=args.beta,
        clip_range=args.clip_range,
    )
    loss_and_grad_fn = ms.value_and_grad(
        net_with_loss, grad_position=None, weights=optimizer.parameters
    )

    gradient_accumulation_steps = args.gradient_accumulation_steps * num_train_timesteps

    for epoch in range(first_epoch, args.num_epochs):
        if epoch % args.eval_freq == 0 and is_main_process:
            outdir = os.path.join(output_dir, "visual", f"epoch_{epoch}")
            # NOTE: DictIterator 会被 consume；每次 eval 都需要重新创建
            test_iter = test_dataloader.create_dict_iterator(output_numpy=True)
            evaluate(
                pipeline,
                reward_fn,
                test_iter,
                sample_neg_prompt_embeds,
                sample_neg_pooled_prompt_embeds,
                scheduler=original_scheduler,
                outdir=outdir,
                ema=ema,
                total_num=len(test_dataloader),
                eval_num_steps=args.eval_num_steps,
                guidance_scale=args.guidance_scale,
                resolution=args.resolution,
                max_sequence_length=args.max_sequence_length,
                seed=args.seed,
            )
        if epoch % args.save_freq == 0 and epoch > 0 and is_main_process:
            save_checkpoint(
                trainable_parameters, outdir=os.path.join(output_dir, "ckpt")
            )
        dist.barrier()

        pipeline.transformer.set_train(False)
        samples = []
        for i in trange(
            args.num_batches_per_epoch,
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process,
            dynamic_ncols=True,
            position=0,
        ):
            prompts = next(train_iter)["prompt"].tolist()

            prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                prompts,
                None,
                None,
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_sequence_length,
            )

            output = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                output_type="pil",
                return_dict=True,
                height=args.resolution,
                width=args.resolution,
            )

            latents = mint.stack(output.all_latents, dim=1).numpy()

            log_probs = mint.stack(output.all_log_probs, dim=1).numpy()

            timesteps = mint.tile(
                pipeline.scheduler.timesteps, (args.train_batch_size, 1)
            ).numpy()

            rewards = reward_fn(output.images, prompts)

            samples.append(
                {
                    "prompts": prompts,
                    "prompt_embeds": prompt_embeds.numpy(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.numpy(),
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        samples = {
            k: (
                np.concatenate([s[k] for s in samples], axis=0)
                if not isinstance(samples[0][k], dict)
                else {
                    sub_key: np.concatenate([s[k][sub_key] for s in samples], axis=0)
                    for sub_key in samples[0][k]
                }
            )
            for k in samples[0].keys()
        }

        reward_avg = samples["rewards"]["avg"][..., None]
        samples["rewards"]["avg"] = np.repeat(reward_avg, args.num_steps, axis=1)

        gathered_rewards = {
            key: gather(ms.tensor(value, dtype=ms.float32)).numpy()
            for key, value in samples["rewards"].items()
        }

        if args.per_prompt_stat_tracking:
            prompts = gather(samples["prompts"].tolist())
            advantages = stat_tracker.update(prompts, gathered_rewards["avg"])
            if len(set(prompts)) != samples_per_epoch // args.num_image_per_prompt:
                pass
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
                gathered_rewards["avg"].std() + 1e-4
            )

        samples["advantages"] = advantages.reshape(
            num_processes, -1, advantages.shape[-1]
        )[process_index]

        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        _ = num_timesteps

        for inner_epoch in range(args.num_inner_epochs):
            perm = np.random.permutation(total_batch_size)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(
                    -1, total_batch_size // args.num_batches_per_epoch, *v.shape[1:]
                )
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.set_train(True)
            train_timesteps = list(range(num_train_timesteps))
            grad_accumulated = None

            loss_scaler = (
                ms.Tensor(1 / gradient_accumulation_steps)
                if gradient_accumulation_steps > 1
                else None
            )

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                disable=not is_main_process,
                dynamic_ncols=True,
                position=0,
            ):
                if args.guidance_scale > 1.0:
                    embeds = mint.cat(
                        [train_neg_prompt_embeds, ms.tensor(sample["prompt_embeds"])]
                    )
                    pooled_embeds = mint.cat(
                        [
                            train_neg_pooled_prompt_embeds,
                            ms.tensor(sample["pooled_prompt_embeds"]),
                        ]
                    )
                else:
                    embeds = ms.tensor(sample["prompt_embeds"])
                    pooled_embeds = ms.tensor(sample["pooled_prompt_embeds"])

                avg_loss = list()
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    leave=False,
                    disable=not is_main_process,
                    dynamic_ncols=True,
                    position=1,
                ):
                    latents = ms.tensor(sample["latents"][:, j])
                    next_latents = ms.tensor(sample["next_latents"][:, j])
                    timesteps = ms.tensor(sample["timesteps"][:, j])
                    advantages = ms.tensor(sample["advantages"][:, j])
                    sample_log_probs = ms.tensor(sample["log_probs"][:, j])

                    step_index = [
                        pipeline.scheduler.index_for_timestep(t) for t in timesteps
                    ]
                    next_step_index = [step + 1 for step in step_index]
                    sigma = pipeline.scheduler.sigmas[step_index].view(-1, 1, 1, 1)
                    sigma_next = pipeline.scheduler.sigmas[next_step_index].view(
                        -1, 1, 1, 1
                    )

                    with ms._no_grad():
                        _, prev_sample_mean_ref, _ = net_with_loss.compute_log_prob(
                            latents,
                            next_latents,
                            timesteps,
                            embeds,
                            pooled_embeds,
                            sigma,
                            sigma_next,
                        )
                    prev_sample_mean_ref = ms.ops.stop_gradient(prev_sample_mean_ref)

                    loss, grad = loss_and_grad_fn(
                        latents,
                        next_latents,
                        timesteps,
                        embeds,
                        pooled_embeds,
                        advantages,
                        sample_log_probs,
                        sigma,
                        sigma_next,
                        prev_sample_mean_ref=prev_sample_mean_ref,
                        loss_scaler=loss_scaler,
                    )

                    if (i * num_train_timesteps + j) % gradient_accumulation_steps == 0:
                        grad_accumulated = grad
                    else:
                        map_(lambda x, y: x.add_(y), grad_accumulated, grad)

                    if (
                        i * num_train_timesteps + j + 1
                    ) % gradient_accumulation_steps == 0:
                        grad_accumulated = tuple(g.contiguous() for g in grad_accumulated)
                        syn_gradients(grad_accumulated)
                        clip_by_global_norm(
                            grad_accumulated, max_norm=args.max_grad_norm
                        )
                        optimizer(grad_accumulated)

                    avg_loss.append(loss.item())
                    global_step += 1

                _ = avg_loss

                if args.ema:
                    ema(trainable_parameters, global_step)


def main():
    parser = argparse.ArgumentParser(
        usage="Training SD3 with GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # =========== general arguments ===========
    group = parser.add_argument_group("general arguments")
    group.add_argument(
        "--reward",
        nargs="+",
        required=True,
        choices=AVAILABLE_SCORERS.keys(),
        help="Reward function(s) to use for training",
    )
    group.add_argument(
        "--reward-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional weights for each reward in --reward (must match length). Defaults to uniform weights.",
    )
    group.add_argument(
        "--resolution",
        default=512,
        type=int,
        help="Image resolution for training and sampling",
    )
    group.add_argument(
        "--max-sequence-length",
        default=128,
        type=int,
        help="Maximum sequence length for text prompts",
    )
    group.add_argument(
        "--model", default=DEFAULT_MODEL, type=str, help="Path to the pretrained model"
    )
    group.add_argument(
        "--run-name",
        type=str,
        help="Name of the run for logging and saving checkpoints",
    )
    group.add_argument(
        "--resume-from", type=str, help="Path to the checkpoint to resume from"
    )
    group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    group.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(Deprecated) Kept for CLI compatibility; has no effect.",
    )

    # ========== training arguments ===========
    group = parser.add_argument_group("training arguments")
    group.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of steps to sample from the diffusion model during training stage",
    )
    group.add_argument(
        "--timestep-fraction",
        type=float,
        default=1.0,
        help="Fraction of timesteps to use for training",
    )
    group.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Mixed precision to use for training",
    )
    group.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the optimizer",
    )
    group.add_argument(
        "--adam-beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    group.add_argument(
        "--adam-beta2", type=float, default=0.999, help="Beta2 for Adam optimizer"
    )
    group.add_argument(
        "--adam-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam optimizer",
    )
    group.add_argument(
        "--adam-epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer"
    )
    group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    group.add_argument(
        "--num-epochs", type=int, default=1000, help="Number of epochs to train for"
    )
    group.add_argument(
        "--num-inner-epochs",
        type=int,
        default=1,
        help="Number of inner epochs to train for",
    )
    group.add_argument(
        "--train-batch-size", type=int, default=2, help="Batch size for training"
    )
    group.add_argument(
        "--num-batches-per-epoch",
        type=int,
        default=12,
        help="Number of batches per epoch",
    )
    group.add_argument(
        "--num-image-per-prompt",
        type=int,
        default=5,
        help="Number of images to generate per prompt",
    )
    group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=6,
        help="Number of gradient accumulation steps",
    )
    group.add_argument(
        "--save-freq",
        type=int,
        default=2,
        help="Frequency of saving checkpoints during training",
    )
    group.add_argument(
        "--beta", type=float, default=0.001, help="KL reward coefficient for training"
    )
    group.add_argument(
        "--adv-clip-max",
        type=float,
        default=5.0,
        help="Maximum value for the advantage clipping",
    )
    group.add_argument(
        "--clip-range", type=float, default=1e-4, help="Clip range for the policy loss"
    )
    group.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use LoRA for training",
    )
    group.add_argument(
        "--lora-path", type=str, default=None, help="Path to the LoRA weights to load"
    )
    group.add_argument(
        "--per-prompt-stat-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to track per-prompt statistics for rewards",
    )
    group.add_argument(
        "--global-std",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use global standard deviation for rewards",
    )

    # ========== sampling arguments ===========
    group = parser.add_argument_group("sampling arguments")
    group.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="Guidance scale for classifier-free guidance",
    )
    group.add_argument(
        "--test-batch-size", type=int, default=5, help="Batch size for evaluation"
    )
    group.add_argument(
        "--eval-freq",
        type=int,
        default=2,
        help="Frequency of evaluation during training",
    )
    group.add_argument(
        "--eval-num-steps",
        type=int,
        default=40,
        help="Number of steps to sample from the diffusion model during evaluation",
    )
    group.add_argument(
        "--validation-num",
        type=int,
        default=10,
        help="Number of validation samples to generate during evaluation",
    )
    group.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use EMA for training",
    )

    # ========== dataset arguments ===========
    group = parser.add_argument_group("dataset arguments")
    group.add_argument(
        "--dataset", type=str, default="dataset/ocr", help="Path to dataset"
    )
    
    # ========== diffusion-rm arguments ===========
    group = parser.add_argument_group("diffusion-rm arguments")
    group.add_argument(
        "--diffusion-rm-checkpoint-path",
        type=str,
        default=None,
        help="Path to Diffusion-RM checkpoint directory",
    )
    group.add_argument(
        "--diffusion-rm-config-path",
        type=str,
        default=None,
        help="Path to Diffusion-RM config file",
    )
    group.add_argument(
        "--diffusion-rm-u",
        type=float,
        default=0.9,
        help="Noise level u for Diffusion-RM (default: 0.9)",
    )
    group.add_argument(
        "--diffusion-rm-pipeline-path",
        type=str,
        default=None,
        help="Pipeline path for Diffusion-RM scorer when not sharing training pipeline. Default: use --model.",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

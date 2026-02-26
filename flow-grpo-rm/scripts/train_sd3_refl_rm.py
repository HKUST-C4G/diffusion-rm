from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
import torch.nn.functional as F
from torchvision import transforms
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.rewards import diffusion_rm_score
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import torch.distributed as dist

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

from accelerate.state import AcceleratorState
import copy

import accelerate.state as _acc_state

# _orig_reset_state = _acc_state.AcceleratorState._reset_state

# def _debug_reset_state(*args, **kwargs):
#     print("\n[DEBUG] AcceleratorState._reset_state() 被调用了，调用栈如下：", file=sys.stderr)
#     traceback.print_stack(file=sys.stderr)
#     print("[DEBUG] ====== 结束 ======\n", file=sys.stderr)
#     return _orig_reset_state(*args, **kwargs)

# _acc_state.AcceleratorState._reset_state = _debug_reset_state

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


class PretrainDataset(Dataset):
    def __init__(self, dataset_path, split='train', resolution=512):
        from datasets import load_dataset, load_from_disk

        # self.dataset = load_dataset(dataset_path, split=split)
        self.dataset = load_from_disk(dataset_path)

        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        pixel_values = self.transform(item['image'].convert('RGB'))
        prompt = item['caption_composition']
        if prompt is None:
            prompt = ""
        else:
            prompt = str(prompt)
        return {"pixel_values": pixel_values, "prompt": prompt}

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        prompts = [example["prompt"] for example in examples]
        return pixel_values, prompts


class InfiniteLoader:
    """
    一个无限循环的 Loader 包装器，
    防止在 while True 训练循环中因为 epoch 结束而报错 StopIteration
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def __iter__(self):
        return self


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = noise_pred_uncond.detach()
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, proxy_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    local_images = []
    local_captions = []
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, all_latents, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
                rewards = executor.submit(reward_fn, all_latents[-1], prompt_embeds, pooled_prompt_embeds)
                proxy_rewards = executor.submit(proxy_reward_fn, images, prompts, prompt_metadata, only_strict=False)
                # yield to to make sure reward computation starts
                time.sleep(0)
                rewards, reward_metadata = rewards.result() # rewards: dict[list]
                proxy_rewards, proxy_reward_metadata = proxy_rewards.result()

        proc_id = accelerator.process_index
        for idx, i in enumerate(range(len(images))):
            image = images[i]
            pil = Image.fromarray(
                (image.detach().float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil = pil.resize((config.resolution, config.resolution))
            
            caption = f"GPU:{proc_id} | {prompts[i]} | {rewards['avg'][i].item():.2f} | {proxy_rewards['avg'][i].item():.2f}"

            local_images.append(pil)
            local_captions.append(caption)

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).detach().float().cpu().numpy()
            all_rewards[key].append(rewards_gather)

        for key, value in proxy_rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).detach().float().cpu().numpy()
            all_rewards[f"proxy_{key}"].append(rewards_gather)

    # gather
    world_size = dist.get_world_size()
    all_images = [None] * world_size
    all_captions = [None] * world_size
    dist.all_gather_object(all_images, local_images)
    dist.all_gather_object(all_captions, local_captions)

    all_images = [item for sublist in all_images for item in sublist]
    all_captions = [item for sublist in all_captions for item in sublist]
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    if accelerator.is_main_process:
        num_samples = min(100, len(all_images))
        # sample_indices = random.sample(range(len(images)), num_samples)
        sample_indices = list(range(num_samples))

        sampled_images = [all_images[i] for i in sample_indices]
        sampled_captions = [all_captions[i] for i in sample_indices]

        images = [wandb.Image(pil, caption=caption) for pil, caption in zip(sampled_images, sampled_captions)]
        accelerator.log({
                "eval_images": images,
                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                **{f"eval_reward_{key}_std": np.std(value[value != -10]) for key, value in all_rewards.items()},
            }, step=global_step,
        )

        # also save on local disk
        save_dir = os.path.join(config.save_eval_dir, f"step_{global_step:04d}")
        os.makedirs(save_dir, exist_ok=True)
        print(len(all_images))
        # TODO: check the test set's num
        for i, image in enumerate(all_images[:num_samples]):
            image.save(os.path.join(save_dir, f"image_{i:04d}.png"))

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if os.path.exists(config.save_dir):
        config.save_dir += f"_{unique_id}"

    if os.path.exists(config.save_eval_dir):
        config.save_eval_dir += f"_{unique_id}"
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.save_eval_dir, exist_ok=True)

    # number of timesteps within each trajectory to train on
    num_train_timesteps = config.sample.num_steps

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        # wandb.init(
        #     project="flow-grpo",
        # )
        accelerator.init_trackers(
            project_name="flow-grpo-rm",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model, torch_dtype=torch.bfloat16
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    # pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    # prepare prompt and reward fn
    # Temporarily disable AcceleratorState reset
    # import pdb; pdb.set_trace()
    original_reset = AcceleratorState._reset_state
    AcceleratorState._reset_state = lambda *args, **kwargs: None
    reward_fn = diffusion_rm_score(pipeline, accelerator.device, inference_dtype)
    proxy_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.proxy_reward_fn)
    
    # eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    AcceleratorState._reset_state = original_reset
    eval_reward_fn = reward_fn
    eval_proxy_reward_fn = proxy_reward_fn
    torch.cuda.empty_cache()
    

    if config.use_lora:
        # Set correct lora layers
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
        if config.train.lora_path:
            if isinstance(pipeline.transformer, PeftModel):
                logger.info(f"Loading Existing LoRA model from {config.train.lora_path}")
                pipeline.transformer.load_adapter(config.train.lora_path, adapter_name="default")

            else:
                logger.info(f"Initializing PeftModel from {config.train.lora_path}...")
                pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
        else:
            if isinstance(pipeline.transformer, PeftModel):
                logger.info(f"Adding new 'default' adapter (rank={transformer_lora_config.r}) into PeftModel...")
                pipeline.transformer.add_adapter("default", transformer_lora_config)
            else:
                logger.info(f"Initializing new PeftModel with 'default' adapter (rank={transformer_lora_config.r})...")
                pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    pipeline.transformer.set_adapter("default")
    transformer = pipeline.transformer
    pipeline_sigmas = pipeline.scheduler.sigmas.to(accelerator.device)
    pipeline_timesteps = pipeline.scheduler.timesteps.to(accelerator.device)
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    if config.train.pretrain_loss > 0.0:
        pretrain_dataset = PretrainDataset(config.train.pretrain_data, 'train', config.resolution)

        raw_pretrain_loader = DataLoader(
             pretrain_dataset,
             batch_size=config.train.pretrain_batch_size,
             shuffle=True,
             collate_fn=pretrain_dataset.collate_fn,
             num_workers=8,
             pin_memory=True,
             drop_last=True
        )


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)
    if config.train.pretrain_loss > 0.0:
        raw_pretrain_loader = accelerator.prepare(raw_pretrain_loader)

        pretrain_iter = InfiniteLoader(raw_pretrain_loader)


    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    # import pdb; pdb.set_trace()
    # print(f"====S1: Finish Loading")    ## 0 -> 33379 MB

    while True:
        if epoch % config.eval_every_n_epochs == 0:
            if accelerator.num_processes > 1:
                eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, eval_proxy_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
                torch.cuda.empty_cache()
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
        train_sampler.set_epoch(epoch)
        prompts, prompt_metadata = next(train_iter)
        # import pdb; pdb.set_trace()

        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        prompt_ids = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)

        # Prepare latent variables
        height = width = config.resolution
        batch_size = prompt_embeds.shape[0]
        num_channels_latents = pipeline.transformer.config.in_channels
        latents = torch.randn(
            (batch_size, num_channels_latents, height // 8, width // 8),
            device=accelerator.device,
        )

        pipeline.scheduler.set_timesteps(config.sample.eval_num_steps, device=accelerator.device)
        timesteps = pipeline.scheduler.timesteps

        mid_timestep = random.randint(30, 39)

        with accelerator.accumulate(pipeline.transformer):
            with autocast():
                if config.train.use_cfg_train:
                    prompt_embeds = torch.cat([sample_neg_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds = torch.cat([sample_neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

                with pipeline.progress_bar(total=mid_timestep) as progress_bar:
                    for i, t in enumerate(timesteps[:mid_timestep]):
                        with torch.no_grad():
                            latent_model_input = torch.cat([latents] * 2) if config.train.use_cfg_train else latents
                            timestep = t.expand(latent_model_input.shape[0]).to(accelerator.device)

                            noise_pred = pipeline.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                return_dict=False,
                            )[0]

                            if config.train.use_cfg_train:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            
                            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                        progress_bar.update()


                latent_model_input = latents
                timestep = timesteps[mid_timestep].expand(latent_model_input.shape[0]).to(accelerator.device)
                # import pdb; pdb.set_trace()
                if config.train.use_cfg_train:
                    _, prompt_embeds= prompt_embeds.chunk(2)
                    _, pooled_prompt_embeds = pooled_prompt_embeds.chunk(2)

                noise_pred = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                sigma_idx = pipeline.scheduler.index_for_timestep(timesteps[mid_timestep])
                sigma = pipeline.scheduler.sigmas[sigma_idx].to(accelerator.device)
                last_latents = latents - sigma * noise_pred

                pred_original_sample = (last_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                pred_original_sample = pred_original_sample.to(dtype=pipeline.vae.dtype)
                # import pdb; pdb.set_trace()
                # print(f"====S2: Finish Rollout Sampling")   ## 33379 -> 35139 MB / 41309 MB
                # print(f"Current Adapter: {pipeline.transformer.active_adapter}")

                with torch.no_grad():     
                    images = pipeline.vae.decode(pred_original_sample, return_dict=False)[0]
                    images = pipeline.image_processor.postprocess(images, output_type="pt") # (B, C, H, W)

                # import pdb; pdb.set_trace()
                # print(f"====S3: Finish VAE Decoding")       ## 35139 -> 41259 MB / 65323 MB

                # compute rewards asynchronously
                with autocast():
                    last_latents = last_latents.to(dtype=inference_dtype)
                    rewards = executor.submit(reward_fn, last_latents, prompt_embeds, pooled_prompt_embeds)
                    proxy_rewards = executor.submit(proxy_reward_fn, images, prompts, prompt_metadata, only_strict=False)
                    # yield to to make sure reward computation starts
                    time.sleep(0)
                    rewards, _ = rewards.result()
                    proxy_rewards, _ = proxy_rewards.result()
                # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()             
                # print(f"====S4: Finish Reward Computation")     ## 41259 -> 41459 MB / 65417 MB

                rewards['avg'] = torch.stack(rewards['avg'], dim=0)
                rewards['proxy_avg'] = torch.stack(proxy_rewards['avg'], dim=0)

                if config.per_prompt_stat_tracking:
                    # gather the prompts across processes
                    gathered_prompts = accelerator.gather(prompts)
                    gathered_rewards = {
                        key: accelerator.gather(value).detach().float().cpu().numpy()
                        for key, value in rewards.items()
                    }
                    gathered_proxy_rewards = {
                        key: accelerator.gather(value).detach().float().cpu().numpy()
                        for key, value in proxy_rewards.items()
                    }
                    advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
                    if accelerator.is_local_main_process:
                        print("len(prompts)", len(prompts))
                        print("len unique prompts", len(set(prompts)))

                    group_size, trained_prompt_num = stat_tracker.get_stats()
                    zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
                else:
                    gathered_rewards = {
                        key: accelerator.gather(value).detach().float().cpu().numpy()
                        for key, value in rewards.items()
                    }
                    reward_std_mean = gathered_rewards['avg'].std()

                proxy_reward_std_mean = accelerator.gather(rewards['proxy_avg']).detach().float().cpu().numpy().std()

                # loss = F.relu(-rewards+2)
                refl_loss = 2-rewards['avg']
                refl_loss = refl_loss.mean()

                loss = refl_loss

                # import pdb; pdb.set_trace()
                if config.train.pretrain_loss > 0.0:
                    # get pretrain batch
                    pretrain_batch = next(pretrain_iter)
                    pretrain_pixel_values, pretrain_prompts = pretrain_batch
                    pretrain_pixel_values = pretrain_pixel_values.to(accelerator.device)

                    # encode pretrain images to latents
                    with torch.no_grad():
                        pretrain_latents = pipeline.vae.encode(
                            pretrain_pixel_values.to(dtype=pipeline.vae.dtype)
                        ).latent_dist.sample()
                        pretrain_latents = (pretrain_latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                        pretrain_latents = pretrain_latents.to(dtype=inference_dtype)

                        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                            pretrain_prompts, 
                            text_encoders, 
                            tokenizers, 
                            max_sequence_length=128, 
                            device=accelerator.device
                        )

                    # sample noise that we'll add to the latents
                    noise = torch.randn_like(pretrain_latents)
                    bsz = pretrain_latents.shape[0]
                    pretrain_timesteps_indices = torch.randint(0, 1000, (bsz,), device=pretrain_latents.device).long()

                    # import pdb; pdb.set_trace()
                    timesteps = pipeline_timesteps[pretrain_timesteps_indices].to(pretrain_latents.device)
                    sigmas = pipeline_sigmas[pretrain_timesteps_indices].to(pretrain_latents.device)
                    sigmas = sigmas[:, None, None, None]
                    
                    noisy_model_input = (1 - sigmas) * pretrain_latents + sigmas * noise

                    model_pred = pipeline.transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                    target = noise - pretrain_latents
                    pretrain_loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                    loss = (loss + config.train.pretrain_loss * pretrain_loss) / (1 + config.train.pretrain_loss)

                # log
                avg_rewards = accelerator.gather(rewards['avg']).detach().float().cpu().numpy().mean()
                avg_loss = accelerator.gather(loss).detach().float().cpu().numpy().mean()
                
                avg_proxy_rewards = accelerator.gather(rewards['proxy_avg']).detach().float().cpu().numpy().mean()
                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "reward": avg_rewards,
                            "reward_std": reward_std_mean,
                            "loss": avg_loss,
                            "epoch": epoch,
                            "proxy_reward": avg_proxy_rewards,
                            "proxy_reward_std": proxy_reward_std_mean,
                        },
                        step=global_step,
                    )
                    print(f"Epoch {epoch}, Global Step {global_step}, Avg Reward: {avg_rewards:.4f}, Avg Loss: {avg_loss:.4f}")
                
                if epoch % 10 == 0:
                    # this is a hack to force wandb to log the images as JPEGs instead of PNGs
                    proc_id = accelerator.process_index

                    num_samples = min(15, len(images))
                    sample_indices = random.sample(range(len(images)), num_samples)

                    local_logged = []
                    for idx, i in enumerate(sample_indices):
                        image = images[i]
                        pil = Image.fromarray(
                            (image.detach().float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        pil = pil.resize((config.resolution, config.resolution))
                        caption = f"GPU:{proc_id} | {prompts[i]} | {rewards['avg'][i].item():.2f}"
                        local_logged.append((pil, caption))

                    world_size = dist.get_world_size()
                    gathered = [None] * world_size
                    dist.all_gather_object(gathered, local_logged)

                    if accelerator.is_main_process:
                        all_logged = [item for sublist in gathered for item in sublist]
                        images = [wandb.Image(pil, caption=caption) for pil, caption in all_logged]
                        accelerator.log({"images": images}, step=global_step)

                # backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), config.train.max_grad_norm
                    )

                    # if epoch % config.eval_freq == 0 and epoch > 0:
                    if epoch % config.eval_freq == 0:
                        if accelerator.num_processes > 1:
                            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, eval_proxy_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
                            torch.cuda.empty_cache()
                    if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
                        save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

                    global_step += 1
                    epoch += 1

                optimizer.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    app.run(main)


import sys
import os
os.umask(0)
from numpy import dtype
import yaml
import argparse
import torch

import sys, logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_cosine_schedule_with_warmup
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
import datetime

from diffusers import StableDiffusion3Pipeline

from diffusion_rm.data.bucket_dataset import create_bucket_dataloader
from diffusion_rm.data.simple_dataset import create_simple_dataloader
from diffusion_rm.models.sd3_rm import SD3RewardModel
from diffusion_rm.utils.vae_utils import VAEProcessor
from diffusion_rm.train.engine import TrainingEngine

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train diffusion reward model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return OmegaConf.create(config)


def create_model_processors(accelerator, config, model_dtype):
    # load pretrained pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.model.backbone_model_id,
        torch_dtype=model_dtype,
    )
    pipeline.vae.to(accelerator.device, dtype=model_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=model_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=model_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=model_dtype)

    pipeline.transformer.to(accelerator.device)


    # Create VAE processor
    vae_processor = VAEProcessor(pipeline.vae)
    noise_scheduler = pipeline.scheduler

    sd3_rm = SD3RewardModel(
        pipeline=pipeline,
        config_model=config.model,
        device=accelerator.device,
        dtype=model_dtype
    )

    # del pipeline  # free memory

    return sd3_rm, vae_processor, noise_scheduler


def get_timesteps_from_u(noise_scheduler, u: torch.Tensor, n_dim: int=4, dtype: torch.dtype=torch.float32):
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps.to(u.device)[indices]

    # get sigmas
    sigmas = noise_scheduler.sigmas.to(u.device)[indices]

    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas, timesteps


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    # prepare output directory
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
    if not config.paths.run_name:
        config.paths.run_name = unique_id
    else:
        config.paths.run_name += f"_{unique_id}"

    config.paths.save_dir = os.path.join(config.paths.save_dir, config.paths.run_name)

    os.makedirs(config.paths.save_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=config.paths.save_dir,
    )


    # Set random seed
    if config.system.seed is not None:
        torch.manual_seed(config.system.seed)
    
    # Setup logging
    if accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,  # 覆盖旧配置，防止重复 handler
        )
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"Starting training with config: {args.config}")
    # Setup W&B
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="diffusion-rm",
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={
                "wandb": {"name": config.paths.run_name},
            }
        )

    model_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    
    # load pretrained pipeline
    logger.info("Loading model processors...")
    sd3_rm, vae_processor, noise_scheduler = create_model_processors(accelerator, config, model_dtype)

    # Create dataloader
    logger.info("Loading datasets...")
    # train_dataset, val_dataset = create_hpd_dataset(config.data)
    # eval_dataset = create_genai_dataset(config.eval_data)
    world_size = accelerator.num_processes
    global_rank = accelerator.process_index


    train_dataloader = create_bucket_dataloader(
        world_size=world_size,
        global_rank=global_rank,
        **config.data.train
    )
    eval_dataloaders = {split: create_simple_dataloader(**cfg) for split, cfg in config.data.eval.items()}

    logger.info("Training dataset size: {}".format(len(train_dataloader.dataset)))
    logger.info(f"Number of training batches per epoch: {len(train_dataloader)}")
    for split, dl in eval_dataloaders.items():
        logger.info(f"Evaluation dataset ({split}) size: {len(dl.dataset)}")
    
    # import pdb; pdb.set_trace()
    
    # Create optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, sd3_rm.parameters()))
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6} M")
    ## for debu, print trainable param names
    # for name, param in sd3_rm.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable parameter: {name}, shape: {param.shape}, dtype: {param.dtype}")

    # import pdb; pdb.set_trace()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=eval(config.training.learning_rate),
        weight_decay=config.training.weight_decay,
    )
    
    # Create LR scheduler
    num_epoch = config.training.num_epochs
    num_training_steps = len(train_dataloader) * num_epoch // accelerator.num_processes // config.training.gradient_accumulation_steps 
    # num_training_steps = 100000
    if config.training.warmup_steps is None:
        warmup_steps = 0
    elif isinstance(config.training.warmup_steps, float):
        warmup_steps = int(num_training_steps * config.training.warmup_steps)
    else:
        warmup_steps = config.training.warmup_steps
    
    if config.training.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        lr_scheduler = None
    
    # Prepare with accelerator
    logger.info("Preparing model, optimizer, and dataloaders with accelerator...")
    sd3_rm.backbone, sd3_rm.reward_head, optimizer, train_dataloader = accelerator.prepare(
        sd3_rm.backbone, sd3_rm.reward_head, optimizer, train_dataloader
    )
    # import pdb; pdb.set_trace()
    
    # def count_params(model, trainable_only=False):
    #     if trainable_only:
    #         params = [p for p in model.parameters() if p.requires_grad]
    #     else:
    #         params = list(model.parameters())
    #     total = sum(p.numel() for p in params)
    #     return total
    # print("Total model params: {:.2f} M".format(count_params(sd3_rm) / 1e6))
    # print("Trainable model params: {:.2f} M".format(count_params(sd3_rm, trainable_only=True) / 1e6))
    # print("backbone params: {:.2f} M".format(count_params(sd3_rm.backbone) / 1e6))
    # print("reward head params: {:.2f} M".format(count_params(sd3_rm.reward_head) / 1e6))
    # import pdb; pdb.set_trace()
    # sd3_rm.backbone = accelerator.prepare(sd3_rm.backbone)
    # sd3_rm.reward_head = accelerator.prepare(sd3_rm.reward_head)
    # optimizer = accelerator.prepare(optimizer)
    # train_dataloader = accelerator.prepare(train_dataloader)

    # TODO: check it
    for split in eval_dataloaders:
        eval_dataloaders[split] = accelerator.prepare(eval_dataloaders[split])
    
    if lr_scheduler:
        lr_scheduler = accelerator.prepare(lr_scheduler)
    
    # Create training engine
    engine = TrainingEngine(
        model=sd3_rm,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        accelerator=accelerator,
        vae_processor=vae_processor,
        noise_scheduler=noise_scheduler,
    )
    engine.get_timesteps_from_u = get_timesteps_from_u
    
    # Resume from checkpoint if specified
    if args.resume:
        raise NotImplementedError("Resuming from checkpoint is not implemented yet.")
        logger.info(f"Resuming from checkpoint: {args.resume}")
        engine.load_checkpoint(args.resume)
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Number of epochs: {num_epoch}")
    logger.info(f"  Number of training steps: {num_training_steps}")
    logger.info(f"  Total Batch size: {config.data.train.base_batch_size * config.training.gradient_accumulation_steps * accelerator.num_processes}")
    logger.info(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    
    progress_bar = tqdm(
        range(num_training_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process
    )
    
    global_step = 0

    for epoch in range(num_epoch):
        train_iter = iter(train_dataloader)

        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                # 当前rank数据耗尽
                logger.info(f"[Rank {accelerator.process_index}] "
                        f"Epoch {epoch+1} 数据耗尽，共处理 {global_step} 个batch")
                break
            # Training step
            train_metrics = engine.train_step(batch)
            # add epoch info
            train_metrics['epoch'] = epoch + 1
            # print(train_metrics)
            
            # Log training metrics
            ## TODO: check if need to sync metrics
            engine.log_metrics(train_metrics, global_step, is_train=True)
            
            if accelerator.sync_gradients:
                # Validation
                if global_step % config.logging.eval_frac == 0 and global_step != 0:
                    logger.info("Running validation...")
                    for split, eval_dataloader in eval_dataloaders.items():
                        logger.info(f"Evaluating on {split} dataset...")
                        val_metrics = engine.validate(eval_dataloader, log_name=f"{split}_val")
                        val_metrics['epoch'] = epoch + 1
                        engine.log_metrics(val_metrics, global_step, is_train=False)
                    
                # Save checkpoint
                if global_step % config.logging.save_frac == 0 and global_step != 0:
                    engine.save_checkpoint(step=global_step)
            
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': str(train_metrics['train/loss']).format(".4f"),
                    'acc': str(train_metrics['train/accuracy']).format(".4f"),
                })
                global_step += 1

        logger.info("Running validation at end of epoch...")
        for split, eval_dataloader in eval_dataloaders.items():
            logger.info(f"Evaluating on {split} dataset...")
            val_metrics = engine.validate(eval_dataloader, log_name=f"{split}_val")
            val_metrics['epoch'] = epoch + 1
            engine.log_metrics(val_metrics, global_step, is_train=False)

        logger.info("Saving checkpoint at end of epoch...")
        engine.save_checkpoint(step=global_step, save_dir=f"epoch_{epoch+1:03d}")
        
        accelerator.wait_for_everyone()
        logger.info(f"***** Finished epoch {epoch+1}/{num_epoch} *****")
    
    # Final validation and save
    # logger.info("Running final validation...")
    # val_metrics = engine.validate(val_dataloader)
    # engine.log_metrics(val_metrics, global_step, is_train=False)
    logger.info("Finishing training...")
    
    # engine.save_checkpoint(is_best=is_best, step=global_step)
    
    
if __name__ == "__main__":
    main()
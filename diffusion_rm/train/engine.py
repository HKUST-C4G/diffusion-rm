"""Training and validation engine."""
import math
import torch
import torch.distributed as dist

import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import json
from safetensors.torch import save_file, load_file


class TrainingEngine:
    """Training engine for the diffusion reward model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        vae_processor=None,
        noise_scheduler = None,
        accelerator=None,
    ):
        self.model = model
        self.vae_processor = vae_processor
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.accelerator = accelerator
        
        # Training state
        self.global_step = 0
        
        # Gradient clipping
        self.max_grad_norm = config.training.max_grad_norm
        
        # Create output directories
        self.output_dir = config.paths.save_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._save_config()
        self.metrics = {}
        self.accumulate_steps = 0

        self.use_ema = config.training.use_ema if hasattr(config.training, 'use_ema') else False
        self.ema_decay = config.training.ema_decay if hasattr(config.training, 'ema_decay') else 0
        self.ema_state = None
        self.original_state = None

        if self.use_ema:
            self._init_ema()

        self.add_noise = config.training.add_noise
        self.uncond_prob = config.training.uncond_prob if hasattr(config.training, 'uncond_prob') else 0.0
        self.model_type = config.model.model_type

        if not config.training.use_vae:
            # Freeze backbone if not using LoRA
            del self.vae_processor
            self.vae_processor = None

    def _init_ema(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema_state = {}
        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if param.requires_grad:
                    self.ema_state[name] = param.clone().detach()

    def _ema_update(self):
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if (not p.requires_grad) or (name not in self.ema_state):
                    continue
                
                ema_param = self.ema_state[name]
                self.ema_state[name] = ema_param * decay + (1.0 - decay) * param.detach()

    def _switch_to_ema(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.original_state = {}
        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if param.requires_grad and name in self.ema_state:
                    self.original_state[name] = param.clone().detach()
                    param.data.copy_(self.ema_state[name].data)
    
    def _switch_to_original(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if param.requires_grad and name in self.original_state:
                    param.data.copy_(self.original_state[name].data)
        self.original_state = None

    
    def _save_config(self):
        """Save configuration to output directory."""
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            config = OmegaConf.to_container(self.config, resolve=True)
            json.dump(config, f, indent=4)
    
    def timestep_sampling(self, weighting_scheme, weighting_scheme_param, batch_size, device='cpu', generator=None, ):
        """Sample timesteps based on the specified weighting scheme."""
        if weighting_scheme == "logit_normal":
            logit_mean = eval(weighting_scheme_param.split('_')[0])
            logit_std = eval(weighting_scheme_param.split('_')[1])

            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            mode_scale = eval(weighting_scheme_param)

            u = torch.rand(size=(batch_size,), device=device, generator=generator)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        elif weighting_scheme == "constant":
            constant_value = eval(str(weighting_scheme_param))

            u = torch.full(size=(batch_size,), fill_value=constant_value, device=device)
        elif weighting_scheme == "uniform":
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
        elif weighting_scheme == "power":
            power = eval(weighting_scheme_param)

            u = torch.rand(size=(batch_size,), device=device, generator=generator) ** power
        elif weighting_scheme == "beta":
            alpha = eval(weighting_scheme_param.split('_')[0])
            beta = eval(weighting_scheme_param.split('_')[1])

            dist = torch.distributions.Beta(alpha, beta)
            u = dist.sample((batch_size,)).to(device)

        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
            
        return u

    @staticmethod
    def get_timesteps_from_u(noise_scheduler, u: torch.Tensor, n_dim: int=4, dtype: torch.dtype=torch.float32):
        raise NotImplementedError("This method should be implemented in train script")

    @staticmethod
    def get_timesteps_from_sigma(noise_scheduler, sigma_target, n_dim=4, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        # sigma_target: (B,) in [0,1]
        sigmas = noise_scheduler.sigmas.to(sigma_target.device)  # (N,)
        # find nearest sigma index
        idx = torch.argmin((sigmas[None, :] - sigma_target[:, None]).abs(), dim=1)
        timesteps = noise_scheduler.timesteps.to(sigma_target.device)[idx]

        sigma = sigmas[idx]
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma, timesteps


    def compute_loss(self, score_chosen, score_reject, sigma=None, eps=1e-6):
        
        if self.model_type == "bt":
            # Bradley-Terry loss
            loss = -nn.functional.logsigmoid(score_chosen - score_reject).mean()
        elif self.model_type == "bt-margin":
            # Bradley-Terry loss with margin
            MARGIN = 1.0
            loss = -nn.functional.logsigmoid(score_chosen - score_reject - MARGIN).mean()
        elif self.model_type == "thurstone":
            # import pdb; pdb.set_trace()
            assert sigma is not None, "Sigma must be provided for Thurstone loss"
            sigma = sigma.reshape(-1, 1).repeat(1, score_chosen.shape[1])  # reshape to match score shape

            scale = 2.0
            min_var = 0.05
            var = scale * sigma ** 2 + min_var
            
            normal_dist = torch.distributions.Normal(0, 1)
            _cur = (score_chosen - score_reject) / ((var + var + eps) ** 0.5)
            p = normal_dist.cdf(_cur)
            ## NLL loss
            # loss = -torch.log(p + eps).mean()
            ## fidelity loss
            loss = 1.0 - torch.sqrt(p + eps)
            loss = loss.mean()
        elif self.model_type == "thurstone-constant":
            # import pdb; pdb.set_trace()
            var = 0.5
            
            normal_dist = torch.distributions.Normal(0, 1)
            _cur = (score_chosen - score_reject) / ((var + var + eps) ** 0.5)
            p = normal_dist.cdf(_cur)
            ## NLL loss
            # loss = -torch.log(p + eps).mean()
            ## fidelity loss
            loss = 1.0 - torch.sqrt(p + eps)
            loss = loss.mean()
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        device = self.accelerator.device

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # import pdb; pdb.set_trace()
        
        # encode prompts
        # import pdb; pdb.set_trace()
        bsz = len(batch['prompt'])
        uncond_prob = torch.rand(bsz, device=device)
        if self.uncond_prob > 0.0:
            mask = (uncond_prob < self.uncond_prob) # (B,)
            mixed_prompts = [("" if mask[i].item() else batch["prompt"][i]) for i in range(bsz)]
            batch['mixed_prompt'] = mixed_prompts
            text_conds = self.model.encode_prompt(batch['mixed_prompt'])

            mask = mask.unsqueeze(-1).float()  # (B, 1)
        else:
            text_conds = self.model.encode_prompt(batch['prompt'])

        if 'latent_chosen' not in batch:
            batch['latent_chosen'] = self.vae_processor.encode(batch['image_chosen'])
        if 'latent_reject' not in batch:
            batch['latent_reject'] = self.vae_processor.encode(batch['image_reject'])


        # Forward pass for both images
        with self.accelerator.accumulate(self.model):
            # random add noise
            noise = torch.randn_like(batch['latent_chosen'])
            bsz = batch['latent_chosen'].shape[0]
            u = self.timestep_sampling(self.config.training.t_weighting_scheme,
                                       self.config.training.t_weighting_scheme_param, bsz, device=device)
            # print(u)
            # sigmas, timesteps = self.get_timesteps_from_u(self.noise_scheduler, u, n_dim=len(noise.shape), dtype=batch['latent_chosen'].dtype)
            sigmas, timesteps = self.get_timesteps_from_sigma(self.noise_scheduler, u, n_dim=len(noise.shape), dtype=batch['latent_chosen'].dtype)
            # import pdb; pdb.set_trace()
            if self.add_noise:
                noisy_model_input = (1.0 - sigmas) * batch['latent_chosen'] + sigmas * noise
            else:
                noisy_model_input = batch['latent_chosen']
            # import pdb; pdb.set_trace()

            # Forward pass for chosen image
            with self.accelerator.autocast():
                scores_chosen = self.model(
                    latents = noisy_model_input,
                    timesteps = timesteps,
                    **text_conds,
                )

            if self.add_noise:
                noisy_model_input = (1.0 - sigmas) * batch['latent_reject'] + sigmas * noise
            else:
                noisy_model_input = batch['latent_reject']
            with self.accelerator.autocast():
                scores_reject = self.model(
                    latents = noisy_model_input,
                    timesteps = timesteps,
                    **text_conds,
                )
            # import pdb; pdb.set_trace()
            # Compute Bradley-Terry loss
            # loss = -nn.functional.logsigmoid(scores_chosen - scores_reject - 1.0).mean()
            loss = self.compute_loss(scores_chosen, scores_reject, sigma=sigmas)
            
            # Backward pass
            if self.accelerator:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # import pdb; pdb.set_trace()
            # Gradient clipping
            if self.max_grad_norm > 0:
                if self.accelerator:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                grad_norm = self._compute_grad_norm()
                
            if not torch.is_tensor(grad_norm):
                grad_norm = torch.tensor(grad_norm, device=device)
            else:
                grad_norm = grad_norm.to(device)
            
            # Optimizer step
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.use_ema and self.accelerator.sync_gradients:
                self._ema_update()

            self.optimizer.zero_grad()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = ((scores_chosen - scores_reject) > 0.0).float().mean()

            metrics = {
                'train/loss': loss.detach(),
                'train/accuracy': accuracy,
                'train/scores_chosen_mean': scores_chosen.detach().mean(),
                'train/scores_reject_mean': scores_reject.detach().mean(),
                'train/grad_norm': grad_norm,
                'train/lr': torch.tensor(self.optimizer.param_groups[0]['lr'], device=device)
            }
            # metrics = {k: torch.tensor(v, device) for k, v in metrics.items()}
            metrics = self.accelerator.reduce(metrics, reduction="mean")
        
        self.global_step += 1
        return metrics
    
    def validate(self, val_dataloader: DataLoader, log_name='val') -> Dict[str, float]:
        """Run validation loop.
        
        Args:
            val_dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        if self.use_ema:
            self._switch_to_ema()

        # selected_u = [0.01, 0.1, 0.5]
        # selected_u = [0.9, 0.8, 0.7]
        selected_u = [0.3, 0.4, 0.5]
            
        total_loss = {u: 0.0 for u in selected_u}
        total_correct = {u: 0.0 for u in selected_u}
        all_scores_chosen = {u: [] for u in selected_u}
        all_scores_reject = {u: [] for u in selected_u}
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", disable=not self.accelerator.is_local_main_process):
                # Move batch to device
                device = self.accelerator.device
                
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass
                text_conds = self.model.encode_prompt(batch['prompt'])
                # print(f"{batch['latent_chosen'].shape} || {batch['latent_reject'].shape}")
                if 'latent_chosen' not in batch:
                    batch['latent_chosen'] = self.vae_processor.encode(batch['image_chosen'])
                if 'latent_reject' not in batch:
                    batch['latent_reject'] = self.vae_processor.encode(batch['image_reject'])

                # Random add noise
                noise = torch.randn_like(batch['latent_chosen'])
                bsz = batch['latent_chosen'].shape[0]
                num_samples += bsz
                
                for u in selected_u:
                    u_tensor = torch.tensor([u] * bsz, device=device)
                    # sigmas, timesteps = self.get_timesteps_from_u(self.noise_scheduler, u_tensor, n_dim=len(noise.shape), dtype=batch['latent_chosen'].dtype)
                    sigmas, timesteps = self.get_timesteps_from_sigma(self.noise_scheduler, u_tensor, n_dim=len(noise.shape), dtype=batch['latent_chosen'].dtype)
                    
                    if self.add_noise:
                        noisy_model_input = (1.0 - sigmas) * batch['latent_chosen'] + sigmas * noise
                    else:
                        noisy_model_input = batch['latent_chosen']

                    # Forward pass for chosen image
                    with self.accelerator.autocast():
                        scores_chosen = self.model(
                            latents=noisy_model_input,
                            timesteps=timesteps,
                            **text_conds,
                        )

                    if self.add_noise:
                        noisy_model_input = (1.0 - sigmas) * batch['latent_reject'] + sigmas * noise
                    else:
                        noisy_model_input = batch['latent_reject']

                    with self.accelerator.autocast():
                        scores_reject = self.model(
                            latents=noisy_model_input,
                            timesteps=timesteps,
                            **text_conds,
                        )

                    # Compute Bradley-Terry loss
                    # loss = -nn.functional.logsigmoid(scores_chosen - scores_reject).sum()
                    loss = self.compute_loss(scores_chosen, scores_reject, sigma=sigmas)
                    total_loss[u] += loss.item()

                    correct = ((scores_chosen - scores_reject) > 0.0).float().sum().item()
                    total_correct[u] += correct
                    
                    all_scores_chosen[u].extend(scores_chosen.cpu().tolist())
                    all_scores_reject[u].extend(scores_reject.cpu().tolist())
        
        # Aggregate metrics
        device = self.accelerator.device
        metrics = {}   
        for u in selected_u:
            global_total_loss = self.accelerator.gather(torch.as_tensor(total_loss[u], device=device)).sum().item()
            global_total_correct = self.accelerator.gather(torch.as_tensor(total_correct[u], device=device)).sum().item()
            global_num_samples = self.accelerator.gather(torch.tensor(num_samples, device=device)).sum().item()
            
            avg_loss = global_total_loss / global_num_samples
            avg_acc = global_total_correct / global_num_samples

            metrics[f'{log_name}_{u}/loss'] = avg_loss
            metrics[f'{log_name}_{u}/accuracy'] = avg_acc


            # collect across different device
            world_size = dist.get_world_size()
            all_scores_chosen_gathered = [None] * world_size
            all_scores_reject_gathered = [None] * world_size
            dist.all_gather_object(all_scores_chosen_gathered, all_scores_chosen[u])
            dist.all_gather_object(all_scores_reject_gathered, all_scores_reject[u])

            all_scores_chosen_gathered = [item for sublist in all_scores_chosen_gathered for item in sublist]
            all_scores_reject_gathered = [item for sublist in all_scores_reject_gathered for item in sublist]


            metrics[f'{log_name}_{u}/scores_chosen_mean'] = torch.tensor(all_scores_chosen_gathered).mean().item()
            metrics[f'{log_name}_{u}/scores_reject_mean'] = torch.tensor(all_scores_reject_gathered).mean().item()
            metrics[f'{log_name}_{u}/scores_chosen_std'] = torch.tensor(all_scores_chosen_gathered).std().item()
            metrics[f'{log_name}_{u}/scores_reject_std'] = torch.tensor(all_scores_reject_gathered).std().item()
        

        if self.use_ema:
            self._switch_to_original()

        return metrics

    def save_checkpoint(self, step: Optional[int] = None, save_dir: Optional[str] = None, ckpt_only=True):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint
            step: Training step (uses global_step if None)
        """
        if self.accelerator.is_main_process:
            if step is None:
                step = self.global_step

            if save_dir is not None:
                save_dir = os.path.join(self.checkpoint_dir, save_dir)
            else:
                save_dir = os.path.join(self.checkpoint_dir, f"step_{step:05d}")
            os.makedirs(save_dir, exist_ok=True)

            # save model state
            if self.config.model.use_lora and not self.config.model.freeze_backbone:
                ## TODO: fix bugs here
                lora_dir = os.path.join(save_dir, f"backbone_lora")
                os.makedirs(lora_dir, exist_ok=True)
                unwrapped_backbone = self.accelerator.unwrap_model(self.model.backbone)
                unwrapped_backbone.save_pretrained(lora_dir)

                unwrapped_reward_head = self.accelerator.unwrap_model(self.model.reward_head)
                torch.save(unwrapped_reward_head.state_dict(), os.path.join(save_dir, "rm_head.pt"))
                # self.model.backbone, self.model.reward_head = self.accelerator.prepare(self.model.backbone, self.model.reward_head)
            elif not self.config.model.freeze_backbone:
                model_state = self.accelerator.unwrap_model(self.model).state_dict()
                torch.save(model_state, os.path.join(save_dir, f"full_model.pt"))
            else:
                ## not using LoRA and freeze backbone, only save reward head
                unwrapped_reward_head = self.accelerator.unwrap_model(self.model.reward_head)
                torch.save(unwrapped_reward_head.state_dict(), os.path.join(save_dir, "rm_head.pt"))

            
            if not ckpt_only:
                checkpoint = {
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': step,
                    'config': self.config
                }

                torch.save(checkpoint, os.path.join(save_dir, "state.pt"))

                if hasattr(self.model, 'ema_state') and self.model.ema_state is not None:
                    torch.save(self.model.ema_state, os.path.join(save_dir, "ema_state.pt"))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        raise NotImplementedError("This method should be implemented in train script")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, is_train=True):
        """Log metrics to wandb and console.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (uses global_step if None)
        """
        if step is None:
            step = self.global_step

        if not is_train:
            self.accelerator.log(metrics, step=step)
            return

        for k, v in metrics.items():
            if k in self.metrics:
                self.metrics[k] += v
            else:
                self.metrics[k] = v
        self.accumulate_steps += 1
        
        # Log to wandb
        if self.accelerator.sync_gradients:
            for k in self.metrics.keys():
                self.metrics[k] /= self.accumulate_steps
            self.accelerator.log(self.metrics, step=step)
            self.metrics = {}
            self.accumulate_steps = 0

    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        return total_norm ** 0.5
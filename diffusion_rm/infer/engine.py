"""Training and validation engine."""
import math
import torch
import torch.distributed as dist

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from accelerate.logging import get_logger
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import json
from safetensors.torch import save_file, load_file

logger = get_logger(__name__)


class InferenceEngine:
    """Inference engine for the diffusion reward model."""
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        vae_processor=None,
        noise_scheduler = None,
        accelerator=None,
        use_vae=False,
    ):
        self.model = model
        self.vae_processor = vae_processor
        self.noise_scheduler = noise_scheduler
        self.config = config
        self.accelerator = accelerator

        self.add_noise = config.training.add_noise if 'training' in config and 'add_noise' in config.training else True
        # self.add_noise = False

        if not use_vae:
            # Freeze backbone if not using LoRA
            del self.vae_processor
            self.vae_processor = None

    @staticmethod
    def get_timesteps_from_u(noise_scheduler, u: torch.Tensor, n_dim: int=4, dtype: torch.dtype=torch.float32):
        raise NotImplementedError("This method should be implemented in train script")

    @staticmethod
    def get_timesteps_from_sigma(noise_scheduler, sigma_target, n_dim=4, dtype=torch.float32):
        # sigma_target: (B,) in [0,1]
        sigmas = noise_scheduler.sigmas.to(sigma_target.device)  # (N,)
        # find nearest sigma index
        idx = torch.argmin((sigmas[None, :] - sigma_target[:, None]).abs(), dim=1)
        timesteps = noise_scheduler.timesteps.to(sigma_target.device)[idx]

        sigma = sigmas[idx]
        while sigma.dim() < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma, timesteps

    @staticmethod
    def calc_pairwise_accuracy(scores: torch.Tensor, pred_ranks: torch.Tensor):
        """Calculate pairwise accuracy given scores and predicted ranks.
        
        Args:
            scores: Tensor of shape [num_imgs]
            pred_ranks: Tensor of shape [num_imgs]
        Returns:
            correct: Number of correct pairwise comparisons
            total: Total number of pairwise comparisons

        """
        num_imgs = len(scores)

        scores_i = scores.unsqueeze(0)  # [1, num_imgs]
        scores_j = scores.unsqueeze(1)  # [num_imgs, 1]

        rank_i = pred_ranks.unsqueeze(0)  # [1, num_imgs]
        rank_j = pred_ranks.unsqueeze(1)  # [num_imgs, 1]

        correct_matrix = ((scores_i > scores_j) & (rank_i < rank_j)) | ((scores_i < scores_j) & (rank_i > rank_j))

        mask = torch.triu(torch.ones(num_imgs, num_imgs, dtype=torch.bool, device=scores.device), diagonal=1) # Upper triangular mask
        correct = correct_matrix.masked_select(mask).sum().item()
        total = mask.sum().item()

        return correct, total

    def validate(self, val_dataloader: DataLoader, csv_path: str=None, cfg_scale=1.0) -> Dict[str, float]:
        """Run validation loop.
        
        Args:
            val_dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
 

        self.model.eval()

        selected_u = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # selected_u = [0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
            
        total_correct = {u: 0 for u in selected_u}
        total_pairs = {u: 0 for u in selected_u}
        total_scores = {u: [] for u in selected_u}
        
        local_detailed_results = []
        
        progress_bar = tqdm(
            val_dataloader, 
            desc="Validating", 
            # leave=False,
            disable=not self.accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                device = self.accelerator.device
                
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                        
                current_sample_result = {
                    "prompt": batch['prompt'][0],
                    "image_paths": batch['image_paths'],
                    "scores": {} # key: u, value: list of scores
                }
                
                # Forward pass
                assert len(batch['prompt']) == 1, "Batch size must be 1 for validation."

                text_conds = self.model.encode_prompt(batch['prompt'])
                latents = [self.vae_processor.encode(batch['images'][i]) for i in range(len(batch['images']))]
                
                uncond_text_conds = self.model.encode_prompt([""] * len(batch['prompt'])) if cfg_scale != 1.0 else None
                
                # Random add noise
                # noise = torch.randn_like(latents[0])
                bsz = latents[0].shape[0]

                for u in selected_u:
                    u_tensor = torch.tensor([u] * bsz, device=device)
                    # sigmas, timesteps = self.get_timesteps_from_u(self.noise_scheduler, u_tensor, n_dim=len(latents[0].shape), dtype=latents[0].dtype)
                    sigmas, timesteps = self.get_timesteps_from_sigma(self.noise_scheduler, u_tensor, n_dim=len(latents[0].shape), dtype=latents[0].dtype)
                    # import pdb; pdb.set_trace()
                    # print(f"u: {u}, sigma: {sigmas[0,0].item()}, timestep: {timesteps[0].item()}")
                    
                    scores = []
                    for i in range(len(batch['images'])):
                        noise = torch.randn_like(latents[i])
                        if self.add_noise:
                            noisy_model_input = (1.0 - sigmas) * latents[i] + sigmas * noise
                        else:
                            noisy_model_input = latents[i]
                        with self.accelerator.autocast():
                            try:
                                if cfg_scale != 1.0:
                                    uncond_text_conds = {
                                        f"uncond_{k}": v for k, v in text_conds.items()
                                    }

                                score = self.model(
                                    latents=noisy_model_input,
                                    timesteps=timesteps,
                                    cfg_scale=cfg_scale,
                                    **text_conds,
                                    **uncond_text_conds if cfg_scale != 1.0 else {},
                                )
                            except Exception as e:
                                print(noisy_model_input.shape)
                                raise e

                        scores.append(score)

                    scores_tensor = torch.cat(scores, dim=1).squeeze(0) # [num_imgs]
                    correct, total = self.calc_pairwise_accuracy(
                        scores=scores_tensor,  # [num_imgs]
                        pred_ranks=batch['rank'].squeeze(0)  # [num_imgs]
                    )
                    # import pdb; pdb.set_trace()
                    total_correct[u] += correct
                    total_pairs[u] += total
                    total_scores[u].append(scores_tensor)  # [num_imgs]
                    
                    current_sample_result["scores"][u] = scores_tensor.cpu().tolist()

                local_detailed_results.append(current_sample_result)
                
                if self.accelerator.is_local_main_process:
                    # import pdb; pdb.set_trace()
                    postfix_dict = {}
                    for u in selected_u:
                        local_avg_acc = total_correct[u] / total_pairs[u] if total_pairs[u] > 0 else 0
                        postfix_dict[f'Acc_{u}'] = f"{local_avg_acc:.4f}" 
                    
                    progress_bar.set_postfix(postfix_dict)
        
        logger.info(f"[Rank {self.accelerator.process_index}] Finished validation loop, waiting for others...")
        self.accelerator.wait_for_everyone()
        
        logger.info(f"[Rank {self.accelerator.process_index}] All processes synchronized, aggregating metrics...")

        # Aggregate metrics
        device = self.accelerator.device
        metrics = {}   
        for u in selected_u:
            global_total_correct = self.accelerator.gather(torch.as_tensor(total_correct[u], device=device)).sum().item()
            global_total_pairs = self.accelerator.gather(torch.as_tensor(total_pairs[u], device=device)).sum().item()

            avg_acc = global_total_correct / global_total_pairs if global_total_pairs > 0 else 0

            metrics[f'val_{u}/accuracy'] = avg_acc
            logger.info(f"Validation Accuracy (u={u}): {avg_acc:.4f}")

            local_scores = torch.cat(total_scores[u], dim=0).cpu()
            world_size = self.accelerator.num_processes
            all_scores_list = [None] * world_size
            dist.all_gather_object(all_scores_list, local_scores)

            all_scores = torch.cat([s.to(device) for s in all_scores_list])


            metrics[f'val_{u}/mean_score'] = all_scores.mean().item()
            metrics[f'val_{u}/std_score'] = all_scores.std().item()

            logger.info(f"Validation Mean Score (u={u}): {metrics[f'val_{u}/mean_score']:.4f}")
            logger.info(f"Validation Std Score (u={u}): {metrics[f'val_{u}/std_score']:.4f}")

        # Save detailed results only on main process
        if csv_path is not None:
            gathered_results = [None for _ in range(self.accelerator.num_processes)]
            dist.all_gather_object(gathered_results, local_detailed_results)
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:

                flat_results = [item for sublist in gathered_results for item in sublist]
                
                self.save_results_to_csv(flat_results, csv_path, selected_u)
                logger.info(f"Saved detailed validation results to {csv_path}")


        return metrics

    def validate_ensemble(self, val_dataloader: DataLoader, csv_path: str=None, cfg_scale=1.0) -> Dict[str, float]:
        """Run validation loop.
        
        Args:
            val_dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        ensemble_u_list = [[0.2, 0.5, 0.7]]
            
        total_correct = {str(ensemble_u): 0 for ensemble_u in ensemble_u_list}
        total_pairs = {str(ensemble_u): 0 for ensemble_u in ensemble_u_list}
        total_scores = {str(ensemble_u): [] for ensemble_u in ensemble_u_list}
        
        local_detailed_results = []
        
        progress_bar = tqdm(
            val_dataloader, 
            desc="Validating", 
            # leave=False,
            disable=not self.accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                device = self.accelerator.device
                
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                        
                current_sample_result = {
                    "prompt": batch['prompt'][0],
                    "image_paths": batch['image_paths'],
                    "scores": {} # key: u, value: list of scores
                }
                
                # Forward pass
                assert len(batch['prompt']) == 1, "Batch size must be 1 for validation."

                text_conds = self.model.encode_prompt(batch['prompt'])
                latents = [self.vae_processor.encode(batch['images'][i]) for i in range(len(batch['images']))]
                
                uncond_text_conds = self.model.encode_prompt([""] * len(batch['prompt'])) if cfg_scale != 1.0 else None
                # (b, 1) tensor, value 0
                gate_override = torch.zeros(len(batch['prompt']), 1).float().to(device) if cfg_scale != 1.0 else None
                gate_override_mask = torch.ones_like(gate_override) if cfg_scale != 1.0 else None
                
                # Random add noise
                # noise = torch.randn_like(latents[0])
                bsz = latents[0].shape[0]

                for ensemble_u in ensemble_u_list:
                    # u_tensor = torch.tensor([u] * bsz, device=device)
                    # List -> [len(ensemble_u), bsz]
                    ensemble_u_tensor = torch.stack([torch.tensor([u] * bsz, device=device) for u in ensemble_u], dim=0)
                    # reshape to [len(ensemble_u) * bsz]
                    ensemble_u_tensor = ensemble_u_tensor.view(-1)
                    
                    # import pdb; pdb.set_trace()
                    # sigmas, timesteps = self.get_timesteps_from_u(self.noise_scheduler, ensemble_u_tensor, n_dim=len(latents[0].shape), dtype=latents[0].dtype)
                    sigmas, timesteps = self.get_timesteps_from_sigma(self.noise_scheduler, ensemble_u_tensor, n_dim=len(latents[0].shape), dtype=latents[0].dtype)
                    sigmas = sigmas.view(len(ensemble_u), bsz)
                    sigmas = [sigmas[j] for j in range(len(ensemble_u))]
                    timesteps = timesteps.view(len(ensemble_u), bsz)
                    timesteps = [timesteps[j] for j in range(len(ensemble_u))]
                    
                    scores = []
                    for i in range(len(batch['images'])):
                        noise = torch.randn_like(latents[i])
                        if self.add_noise:
                            # noisy_model_input = (1.0 - sigmas) * latents[i] + sigmas * noise
                            noisy_model_inputs = [(1.0 - sigmas[j]) * latents[i] + sigmas[j] * noise for j in range(len(ensemble_u))]
                        else:
                            noisy_model_inputs = [latents[i]] * len(ensemble_u)

                        forward_to_call = self.model.module.forward_ensemble if hasattr(self.model, 'module') else self.model.forward_ensemble
                        with self.accelerator.autocast():
                            try:
                                score = forward_to_call(
                                    latents=noisy_model_inputs,
                                    ensemble_timesteps=timesteps,
                                    **text_conds,
                                )
                                if cfg_scale != 1.0:
                                    uncond_score = forward_to_call(
                                        latents=noisy_model_inputs,
                                        ensemble_timesteps=timesteps,
                                        gate_override=gate_override,
                                        gate_override_mask=gate_override_mask,
                                        **uncond_text_conds,
                                    )
                                    score = uncond_score + cfg_scale * (score - uncond_score)
                            except Exception as e:
                                print(noisy_model_inputs[0].shape)
                                raise e

                        scores.append(score)

                    scores_tensor = torch.cat(scores, dim=1).squeeze(0) # [num_imgs]
                    correct, total = self.calc_pairwise_accuracy(
                        scores=scores_tensor,  # [num_imgs]
                        pred_ranks=batch['rank'].squeeze(0)  # [num_imgs]
                    )

                    total_correct[str(ensemble_u)] += correct
                    total_pairs[str(ensemble_u)] += total
                    total_scores[str(ensemble_u)].append(scores_tensor)  # [num_imgs]
                    
                    current_sample_result["scores"][str(ensemble_u)] = scores_tensor.cpu().tolist()
                local_detailed_results.append(current_sample_result)
                
                if self.accelerator.is_local_main_process:
                    # import pdb; pdb.set_trace()
                    postfix_dict = {}
                    for ensemble_u in ensemble_u_list:
                        local_avg_acc = total_correct[str(ensemble_u)] / total_pairs[str(ensemble_u)] if total_pairs[str(ensemble_u)] > 0 else 0
                        postfix_dict[f'Acc_{ensemble_u}'] = f"{local_avg_acc:.4f}" 
                    
                    progress_bar.set_postfix(postfix_dict)
        
        logger.info(f"[Rank {self.accelerator.process_index}] Finished validation loop, waiting for others...")
        self.accelerator.wait_for_everyone()
        
        logger.info(f"[Rank {self.accelerator.process_index}] All processes synchronized, aggregating metrics...")

        # Aggregate metrics
        device = self.accelerator.device
        metrics = {}   
        for ensemble_u in ensemble_u_list:
            global_total_correct = self.accelerator.gather(torch.as_tensor(total_correct[str(ensemble_u)], device=device)).sum().item()
            global_total_pairs = self.accelerator.gather(torch.as_tensor(total_pairs[str(ensemble_u)], device=device)).sum().item()

            avg_acc = global_total_correct / global_total_pairs if global_total_pairs > 0 else 0

            metrics[f'val_{ensemble_u}/accuracy'] = avg_acc
            logger.info(f"Validation Accuracy (u={ensemble_u}): {avg_acc:.4f}")

            local_scores = torch.cat(total_scores[str(ensemble_u)], dim=0).cpu()
            world_size = self.accelerator.num_processes
            all_scores_list = [None] * world_size
            dist.all_gather_object(all_scores_list, local_scores)

            all_scores = torch.cat([s.to(device) for s in all_scores_list])


            metrics[f'val_{ensemble_u}/mean_score'] = all_scores.mean().item()
            metrics[f'val_{ensemble_u}/std_score'] = all_scores.std().item()

            logger.info(f"Validation Mean Score (u={ensemble_u}): {metrics[f'val_{ensemble_u}/mean_score']:.4f}")
            logger.info(f"Validation Std Score (u={ensemble_u}): {metrics[f'val_{ensemble_u}/std_score']:.4f}")
        # Save detailed results only on main process
        if csv_path is not None:
            gathered_results = [None for _ in range(self.accelerator.num_processes)]
            dist.all_gather_object(gathered_results, local_detailed_results)
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:

                flat_results = [item for sublist in gathered_results for item in sublist]
                
                ensemble_u_list = [str(ensemble_u) for ensemble_u in ensemble_u_list]
                self.save_results_to_csv(flat_results, csv_path, ensemble_u_list)
                logger.info(f"Saved detailed validation results to {csv_path}")


        return metrics

    def save_results_to_csv(self, results: list, csv_path: str, selected_u: list):
        import pandas as pd
        rows = []
        for item in results:
            row = {
                "prompt": item["prompt"],
            }
            
            for idx, path in enumerate(item["image_paths"]):
                row[f"image_path_{idx}"] = path
                
            for u in selected_u:
                scores = item["scores"][u]
                for idx, score in enumerate(scores):
                    row[f"score_u{u}_img{idx}"] = score
                    
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        cols = df.columns.tolist()
        cols.sort(key=lambda x: (
            0 if x == "prompt" else
            1 if x.startswith("image_path_") else
            2 if x.startswith("score_") else
            3
        ))
        
        df = df[cols]
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed validation results to {csv_path}")
        

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if self.config.model.use_lora:
            # Load LoRA weights
            lora_dir = os.path.join(checkpoint_path, "backbone_lora")
            if os.path.exists(lora_dir):
                logger.info(f"Loading LoRA weights from: {lora_dir}")
                if self.accelerator:
                    unwrapped_backbone = self.accelerator.unwrap_model(self.model.backbone)
                    unwrapped_backbone.load_adapter(lora_dir, adapter_name="rm_lora")
                    unwrapped_backbone.set_adapter("rm_lora")
                else:
                    self.model.backbone.load_adapter(lora_dir, adapter_name="rm_lora")
                    self.model.backbone.set_adapter("rm_lora")
            else:
                logger.info(f"Warning: LoRA directory not found: {lora_dir}")

            # Load reward head
            rm_head_path = os.path.join(checkpoint_path, "rm_head.pt")
            if os.path.exists(rm_head_path):
                logger.info(f"Loading reward head from: {rm_head_path}")
                reward_head_state = torch.load(rm_head_path, map_location=self.accelerator.device)
                if self.accelerator:
                    unwrapped_reward_head = self.accelerator.unwrap_model(self.model.reward_head)
                    unwrapped_reward_head.load_state_dict(reward_head_state)
                else:
                    self.model.reward_head.load_state_dict(reward_head_state)
            else:
                logger.info(f"Warning: Reward head file not found: {rm_head_path}")
                
        elif not self.config.model.freeze_backbone:
            # Load full model
            full_model_path = os.path.join(checkpoint_path, "full_model.pt")
            if os.path.exists(full_model_path):
                logger.info(f"Loading full model from: {full_model_path}")
                model_state = torch.load(full_model_path, map_location=self.accelerator.device)
                if self.accelerator:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    unwrapped_model.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(model_state)
            else:
                logger.info(f"Warning: Full model file not found: {full_model_path}")
        
        else:
            # Load only reward head (backbone is frozen)
            rm_head_path = os.path.join(checkpoint_path, "rm_head.pt")
            if os.path.exists(rm_head_path):
                logger.info(f"Loading reward head from: {rm_head_path}")
                reward_head_state = torch.load(rm_head_path, map_location=self.accelerator.device)
                if self.accelerator:
                    unwrapped_reward_head = self.accelerator.unwrap_model(self.model.reward_head)
                    unwrapped_reward_head.load_state_dict(reward_head_state)
                else:
                    self.model.reward_head.load_state_dict(reward_head_state)
            else:
                logger.info(f"Warning: Reward head file not found: {rm_head_path}")
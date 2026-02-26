"""Diffusion-based reward model."""

import torch

import torch.nn as nn
from typing import Dict, Any, Optional, List
from diffusers import DiffusionPipeline
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
import warnings

from diffusers import QwenImagePipeline

from .reward_head import RewardHead



class QwenImageBackbone(nn.Module):
    def __init__(self, transformer, config_model):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed

        self.txt_norm = transformer.txt_norm
        
        self.img_in = transformer.img_in
        self.txt_in = transformer.txt_in

        # import pdb; pdb.set_trace()
        self.transformer_blocks = nn.ModuleList(
            transformer.transformer_blocks[:config_model.num_transformer_layers]    # total 60 layers
        )

        self.visual_head_idx = config_model.visual_head_idx
        self.text_head_idx = config_model.text_head_idx
        
        # import pdb; pdb.set_trace()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_shapes: torch.Tensor = None,
        txt_seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        temb = self.time_text_embed(timestep, hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        hidden_states_list = [hidden_states] if self.visual_head_idx[0] == 0 else []
        encoder_hidden_states_list = [encoder_hidden_states] if self.text_head_idx[0] == 0 else []
        for index_block, block in enumerate(self.transformer_blocks):            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            
            if index_block + 1 in self.visual_head_idx:
                hidden_states_list.append(hidden_states)
            if index_block + 1 in self.text_head_idx:
                encoder_hidden_states_list.append(encoder_hidden_states)

        return temb, hidden_states_list, encoder_hidden_states_list
        


class QwenImageRewardModel(nn.Module):
    """Diffusion-based reward model using pretrained transformer backbone."""
    
    def __init__(self, pipeline, config_model, device, dtype, vae_scale_factor):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        text_encoder = pipeline.text_encoder
        text_encoder.requires_grad_(False)

        self.text_encoding_pipeline = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path="Qwen/Qwen-Image",
            vae=None,
            transformer=None,
            tokenizer=pipeline.tokenizer,
            text_encoder=text_encoder,
            scheduler=None,
        )

        # use only the first N layers of the transformer
        self.backbone = QwenImageBackbone(
            transformer=pipeline.transformer,
            config_model=config_model,
        )

        if config_model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False        
        elif config_model.use_lora and config_model.lora_config is not None:
            # Apply LoRA if specified
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            ]
            exclude_modules = [
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.add_q_proj",
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.add_k_proj",
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.add_v_proj",
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.to_add_out",
            ]
            if config_model.use_text_features and config_model.text_head_idx[-1] == config_model.num_transformer_layers:
                exclude_modules = None
            
            lora_config = LoraConfig(
                r = config_model.lora_config.r,
                lora_alpha = config_model.lora_config.lora_alpha,
                init_lora_weights = config_model.lora_config.init_lora_weights,
                target_modules = target_modules,
                exclude_modules = exclude_modules,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.to(device, dtype=dtype)
            def list_lora_module_paths_from_params(model, only_trainable=False):
                paths = set()
                for n, p in model.named_parameters():
                    if "lora_" in n and (not only_trainable or p.requires_grad):
                        # 形如 "...attn.to_q.lora_A.default.weight" -> 截掉从 ".lora_" 开始的后缀
                        paths.add(n.split(".lora_")[0])
                return sorted(paths)

            # 用法：
            # print("\n".join(list_lora_module_paths_from_params(self.backbone)))
            # print("\n".join(list_lora_module_paths_from_params(self.backbone, only_trainable=True)))
            
            # import pdb; pdb.set_trace()
            
        # Get transformer output dimension
        backbone_dim = pipeline.transformer.inner_dim
        # Initialize reward head
        self.reward_head = RewardHead(
            token_dim=backbone_dim,
            n_visual_heads=len(config_model.visual_head_idx),
            n_text_heads=len(config_model.text_head_idx),
            patch_size=pipeline.transformer.config.patch_size,
            t_embed_dim=backbone_dim,
            use_t_embed=config_model.use_t_embed,
            # patch_size=2,
            **config_model.reward_head
        )

        self.reward_head = self.reward_head.to(device, dtype=dtype)

        self.vae_scale_factor = vae_scale_factor

        self.use_logistic = config_model.use_logistic if hasattr(config_model, 'use_logistic') else False
        if self.use_logistic:
            self.eta1 = 2.0
            self.eta2 = -2.0
            self.eta3 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.eta4 = nn.Parameter(torch.tensor(0.15), requires_grad=True)

    def encode_prompt(self, prompts, max_sequence_length=512):
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.text_encoding_pipeline.encode_prompt(
                prompt=prompts, max_sequence_length=max_sequence_length
            )

        return {
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_embeds_mask,
        }
    
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents
    

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _logistic(self, x):
        if not self.use_logistic:
            return x
        
        exp_pow = -1 * (x - self.eta3) / (torch.abs(self.eta4) + 1e-6)
        return (self.eta1 - self.eta2) / (1 + torch.exp(exp_pow)) + self.eta2
        
    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        timesteps: torch.LongTensor,
        gate_override=None,
        gate_override_mask=None,
    ) -> Dict[str, torch.Tensor]:
        # import pdb; pdb.set_trace()
        # latents = latents.permute(0, 2, 1, 3, 4)

        b, c, _, h, w = latents.shape
        latents = self._pack_latents(latents, b, c, h, w)

        img_shapes = [
            (1, h // 2, w // 2)
        ] * b

        # import pdb; pdb.set_trace()
        ## check timesteps, should between [0, 1]

        temb, hidden_states_list, encoder_hidden_states_list = self.backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timesteps / 1000.0,
            img_shapes=img_shapes,
            txt_seq_lens=encoder_hidden_states_mask.sum(dim=1).tolist(),
        )
        reward = self.reward_head(
            visual_features=hidden_states_list,
            text_features=encoder_hidden_states_list,
            t_embed=temb,
            hw=(h, w),
            gate_override=gate_override,
            gate_override_mask=gate_override_mask,
        )

        if self.use_logistic:
            reward = self._logistic(reward)
            
        return reward

        
    
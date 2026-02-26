"""Diffusion-based reward model."""

import torch

import torch.nn as nn
from typing import Dict, Any, Optional, List
from diffusers import DiffusionPipeline
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
import warnings

from .reward_head import RewardHead


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


class FLUXBackbone(nn.Module):
    def __init__(self, transformer, config_model):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder
        self.x_embedder = transformer.x_embedder

        self.transformer_blocks = nn.ModuleList(
            transformer.transformer_blocks[:config_model.num_transformer_layers]
        )

        self.visual_head_idx = config_model.visual_head_idx
        self.text_head_idx = config_model.text_head_idx
        
        # import pdb; pdb.set_trace()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, guidance, pooled_projections)   # [0, 1000]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)

        image_rotary_emb = self.pos_embed(ids)

        hidden_states_list = [hidden_states] if self.visual_head_idx[0] == 0 else []
        encoder_hidden_states_list = [encoder_hidden_states] if self.text_head_idx[0] == 0 else []
        for index_block, block in enumerate(self.transformer_blocks):            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            
            if index_block + 1 in self.visual_head_idx:
                hidden_states_list.append(hidden_states)
            if index_block + 1 in self.text_head_idx:
                encoder_hidden_states_list.append(encoder_hidden_states)

        return temb, hidden_states_list, encoder_hidden_states_list
        


class FLUXRewardModel(nn.Module):
    """Diffusion-based reward model using pretrained transformer backbone."""
    
    def __init__(self, pipeline, config_model, device, dtype, vae_scale_factor):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2

        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        self.text_encoders = [text_encoder_1, text_encoder_2]
        self.tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

        # use only the first N layers of the transformer
        self.backbone = FLUXBackbone(
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
            frozen_layers_indices = [0, 1, 2, 3]
            for idx in frozen_layers_indices:
                exclude_modules.append(f"transformer_blocks.{idx}.attn")


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
            t_embed_dim=backbone_dim,
            use_t_embed=config_model.use_t_embed,
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

    def encode_prompt(self, prompts):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                self.text_encoders, self.tokenizers, prompts, max_sequence_length=128
            )
            prompt_embeds = prompt_embeds.to(self.text_encoders[0].device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.text_encoders[0].device)

        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
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

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def _logistic(self, x):
        if not self.use_logistic:
            return x
        
        exp_pow = -1 * (x - self.eta3) / (torch.abs(self.eta4) + 1e-6)
        return (self.eta1 - self.eta2) / (1 + torch.exp(exp_pow)) + self.eta2
        
    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor],
        txt_ids: torch.Tensor,
        timesteps: torch.LongTensor,
        gate_override=None,
        gate_override_mask=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        b, c, h, w = latents.shape
        latents = self._pack_latents(latents, b, c, h, w)

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size=b,
            height=h // 2,
            width=w // 2,
            device=latents.device,
            dtype=latents.dtype,
        )

        guidance = torch.Tensor([3.5]).to(device=latents.device, dtype=latents.dtype)

        temb, hidden_states_list, encoder_hidden_states_list = self.backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps / 1000.0,
            img_ids=latent_image_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        # import pdb; pdb.set_trace()
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

        
    
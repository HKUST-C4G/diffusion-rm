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
    max_sequence_length,
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
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

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
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


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

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


class SD3Backbone(nn.Module):
    def __init__(self, transformer, config_model):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder

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
        unpatched: bool = False,
    ) -> torch.Tensor:
        
        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)


        hidden_states_list = [hidden_states] if self.visual_head_idx[0] == 0 else []
        encoder_hidden_states_list = [encoder_hidden_states] if self.text_head_idx[0] == 0 else []
        for index_block, block in enumerate(self.transformer_blocks):            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
            # import pdb; pdb.set_trace()
            
            if index_block + 1 in self.visual_head_idx:
                hidden_states_list.append(hidden_states)
            if index_block + 1 in self.text_head_idx:
                encoder_hidden_states_list.append(encoder_hidden_states)

        return temb, hidden_states_list, encoder_hidden_states_list
        


class SD3RewardModel(nn.Module):
    """Diffusion-based reward model using pretrained transformer backbone."""
    
    def __init__(self, pipeline, config_model, device, dtype):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_3 = pipeline.text_encoder_3

        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        text_encoder_3.requires_grad_(False)

        self.text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]
        self.tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

        # use only the first N layers of the transformer
        self.backbone = SD3Backbone(
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
            # import pdb; pdb.set_trace()
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.to(device, dtype=dtype)
            def list_lora_module_paths_from_params(model, only_trainable=False):
                paths = set()
                for n, p in model.named_parameters():
                    if "lora_" in n and (not only_trainable or p.requires_grad):
                        # 形如 "...attn.to_q.lora_A.default.weight" -> 截掉从 ".lora_" 开始的后缀
                        paths.add(n.split(".lora_")[0])
                return sorted(paths)

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
            # patch_size=1,
            **config_model.reward_head
        )

        self.reward_head = self.reward_head.to(device, dtype=dtype)
        
        self.use_logistic = config_model.use_logistic if hasattr(config_model, 'use_logistic') else False
        if self.use_logistic:
            self.eta1 = 2.0
            self.eta2 = -2.0
            self.eta3 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.eta4 = nn.Parameter(torch.tensor(0.15), requires_grad=True)
    
    def _logistic(self, x):
        if not self.use_logistic:
            return x
        
        exp_pow = -1 * (x - self.eta3) / (torch.abs(self.eta4) + 1e-6)
        return (self.eta1 - self.eta2) / (1 + torch.exp(exp_pow)) + self.eta2

    def encode_prompt(self, prompts):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                self.text_encoders, self.tokenizers, prompts, max_sequence_length=128
            )
            prompt_embeds = prompt_embeds.to(self.text_encoders[0].device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.text_encoders[0].device)

        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds
        }


    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor],
        timesteps: torch.LongTensor,
        **kwargs,
    ):
        b, c, h, w = latents.shape
        temb, hidden_states_list, encoder_hidden_states_list = self.backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps,
            unpatched=False,
        )


        reward = self.reward_head(
            visual_features=hidden_states_list,
            text_features=encoder_hidden_states_list,
            t_embed=temb,
            hw=(h, w),
        )
        # import pdb; pdb.set_trace()
        if self.use_logistic:
            reward = self._logistic(reward)
            
        return reward

    def forward_ensemble(
        self,
        latents: List[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor],
        ensemble_timesteps: List[torch.LongTensor],
    ):
        b, c, h, w = latents[0].shape
        ensemble_temb = []
        ensemble_hidden_states = []
        ensemble_encoder_hidden_states = []
        for i, timesteps in enumerate(ensemble_timesteps):
            temb, hidden_states_list, encoder_hidden_states_list = self.backbone(
                hidden_states=latents[i],
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timesteps,
                unpatched=False,
            )
            ensemble_temb.append(temb)
            ensemble_hidden_states.append(hidden_states_list)
            ensemble_encoder_hidden_states.append(encoder_hidden_states_list)
        
        ensemble_temb = torch.stack(ensemble_temb, dim=1)  # (B, T, D)

        if hasattr(self.reward_head, 'module'):
            reward = self.reward_head.module.forward_ensemble(
                visual_features_per_t=ensemble_hidden_states,
                text_features_per_t=ensemble_encoder_hidden_states,
                t_embed_per_t=ensemble_temb,
            )
        else:
            reward = self.reward_head.forward_ensemble(
                visual_features_per_t=ensemble_hidden_states,
                text_features_per_t=ensemble_encoder_hidden_states,
                t_embed_per_t=ensemble_temb,
            )
        # import pdb; pdb.set_trace()
        if self.use_logistic:
            reward = self._logistic(reward)
            
        return reward


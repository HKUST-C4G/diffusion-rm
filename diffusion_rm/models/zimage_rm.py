"""Diffusion-based reward model."""

import torch

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, Optional, List
from diffusers import DiffusionPipeline
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
import warnings


from .reward_head import RewardHead
from .zimage_transformers import ZImageBackbone



def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: str,
    max_sequence_length=512,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    device = device if device is not None else text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_list = []

    for i in range(len(prompt_embeds)):
        prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

    return prompt_embeds_list


class ZImageRewardModel(nn.Module):
    """Diffusion-based reward model using pretrained transformer backbone."""
    
    def __init__(self, pipeline, config_model, device, dtype):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        text_encoder = pipeline.text_encoder

        text_encoder.requires_grad_(False)

        self.text_encoder = text_encoder
        self.tokenizer = pipeline.tokenizer

        # use only the first N layers of the transformer
        self.backbone = ZImageBackbone(
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
            ]
            # exclude_modules = [
            #     "noise_refiner.0.attention.to_q",
            #     "noise_refiner.0.attention.to_k",
            #     "noise_refiner.0.attention.to_v",
            #     "noise_refiner.0.attention.to_out.0",
            #     "noise_refiner.1.attention.to_q",
            #     "noise_refiner.1.attention.to_k",
            #     "noise_refiner.1.attention.to_v",
            #     "noise_refiner.1.attention.to_out.0",
            #     "context_refiner.0.attention.to_q",
            #     "context_refiner.0.attention.to_k",
            #     "context_refiner.0.attention.to_v",
            #     "context_refiner.0.attention.to_out.0",
            #     "context_refiner.1.attention.to_q",
            #     "context_refiner.1.attention.to_k",
            #     "context_refiner.1.attention.to_v",
            #     "context_refiner.1.attention.to_out.0",
            # ]
            exclude_modules = []
            # if config_model.use_text_features and config_model.text_head_idx[-1] == config_model.num_transformer_layers:
            #     exclude_modules = None
            # frozen_layers_indices = [0, 1, 2, 3]
            # for layer_idx in frozen_layers_indices:
            #     exclude_modules.extend([
            #         f"layers.{layer_idx}.attention.to_q",
            #         f"layers.{layer_idx}.attention.to_k",
            #         f"layers.{layer_idx}.attention.to_v",
            #         f"layers.{layer_idx}.attention.to_out.0",
            #     ])
            
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

            print("\n".join(list_lora_module_paths_from_params(self.backbone)))
            print("\n".join(list_lora_module_paths_from_params(self.backbone, only_trainable=True)))
            
            # import pdb; pdb.set_trace()
            
        # Get transformer output dimension
        backbone_dim = pipeline.transformer.dim
        # import pdb; pdb.set_trace()
        # Initialize reward head
        self.reward_head = RewardHead(
            token_dim=backbone_dim,
            n_visual_heads=len(config_model.visual_head_idx),
            n_text_heads=len(config_model.text_head_idx),
            t_embed_dim=256,
            use_t_embed=config_model.use_t_embed,
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
            prompt_embeds = encode_prompt(
                self.text_encoder, self.tokenizer, prompts, max_sequence_length=128
            )
            # prompt_embeds = prompt_embeds.to(self.text_encoder.device)
            prompt_embeds = [prompt_embed.to(self.text_encoder.device) for prompt_embed in prompt_embeds]

        return {
            "encoder_hidden_states": prompt_embeds,
        }


    def forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: List[torch.Tensor],
        timesteps: torch.LongTensor,
        uncond_encoder_hidden_states: Optional[torch.Tensor] = None,
        uncond_pooled_projections: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        gate_override=None,
        gate_override_mask=None,
        **kwargs,
    ):
        # import pdb; pdb.set_trace()
        timestep_model_input = (1000 - timesteps) / 1000.0  # convert to diffusion timestep

        b, c, h, w = latents.shape

        latent_model_input = latents.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))


        temb, hidden_states_list, encoder_hidden_states_list = self.backbone(
            x=latent_model_input_list,
            cap_feats=encoder_hidden_states,
            t=timestep_model_input,
        )
        # import pdb; pdb.set_trace()
        if cfg_scale != 0.0 and uncond_encoder_hidden_states is not None:
            raise NotImplementedError("CFG with uncond_encoder_hidden_states not implemented yet.")
            _, uncond_hidden_states_list, uncond_encoder_hidden_states_list = self.backbone(
                hidden_states=latents,
                encoder_hidden_states=uncond_encoder_hidden_states,
                pooled_projections=uncond_pooled_projections,
                timestep=timesteps,
                unpatched=False,
            )
            # CFG: scale the difference between conditional and unconditional features
            hidden_states_list = [
                uncond_h + cfg_scale * (cond_h - uncond_h)
                for uncond_h, cond_h in zip(uncond_hidden_states_list, hidden_states_list)
            ]
            encoder_hidden_states_list = [
                uncond_e + cfg_scale * (cond_e - uncond_e)
                for uncond_e, cond_e in zip(uncond_encoder_hidden_states_list, encoder_hidden_states_list)
            ]


        reward = self.reward_head(
            visual_features=hidden_states_list,
            text_features=encoder_hidden_states_list,
            t_embed=temb,
            hw=(h, w),
            gate_override=gate_override,
            gate_override_mask=gate_override_mask,
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
        gate_override=None,
        gate_override_mask=None,
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


"""Diffusion-based reward model."""

import mindspore as ms
from mindspore import mint, nn

from typing import Optional
import numpy as np
from diffusers import DiffusionPipeline
from transformers import AutoConfig
from mindone.peft import LoraConfig, get_peft_model

from .reward_head import RewardHeadV3


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
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
            return_tensors="np",
        )
        text_input_ids = ms.Tensor(np.asarray(text_inputs.input_ids), dtype=ms.int32)
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids)[0]

    dtype = text_encoder.dtype
    # prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
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
            return_tensors="np",
        )

        text_input_ids = ms.Tensor(np.asarray(text_inputs.input_ids), dtype=ms.int32)
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    outputs = text_encoder(text_input_ids, output_hidden_states=True)
    pooled_prompt_embeds = outputs[0]
    last_hidden_state = outputs[1]
    hidden_states = outputs[2] if len(outputs) > 2 else None
    prompt_embeds = hidden_states[-2] if hidden_states is not None else last_hidden_state

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view((batch_size * num_images_per_prompt, seq_len, -1))

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
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
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = mint.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = mint.cat(clip_pooled_prompt_embeds_list, dim=-1)
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
    )

    clip_prompt_embeds = mint.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = mint.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


class SD3Backbone(nn.Cell):
    def __init__(self, transformer, config_model):
        super().__init__()
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder

        self.transformer_blocks = nn.CellList(
            list(transformer.transformer_blocks[:config_model.num_transformer_layers])
        )

        self.visual_head_idx = config_model.visual_head_idx
        self.text_head_idx = config_model.text_head_idx

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        pooled_projections: ms.Tensor = None,
        timestep: ms.Tensor = None,
        unpatched: bool = False,
        **kwargs,
    ) -> ms.Tensor:
        _ = unpatched
        _ = kwargs

        _height, _width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        temb = temb.to(hidden_states.dtype)
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        hidden_states_list = [hidden_states] if self.visual_head_idx[0] == 0 else []
        encoder_hidden_states_list = [encoder_hidden_states] if self.text_head_idx[0] == 0 else []
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

            if index_block + 1 in self.visual_head_idx:
                hidden_states_list.append(hidden_states)
            if index_block + 1 in self.text_head_idx:
                encoder_hidden_states_list.append(encoder_hidden_states)

        return temb, hidden_states_list, encoder_hidden_states_list


class SD3RewardModel(nn.Cell):
    """Diffusion-based reward model using pretrained transformer backbone."""

    def __init__(self, pipeline, config_model, dtype):
        super().__init__()
        self.config_model = config_model
        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_3 = pipeline.text_encoder_3

        text_encoder_1.requires_grad = False
        text_encoder_2.requires_grad = False
        text_encoder_3.requires_grad = False

        self.text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]
        self.tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

        self.backbone = SD3Backbone(
            transformer=pipeline.transformer,
            config_model=config_model,
        )

        if config_model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif config_model.use_lora and config_model.lora_config is not None:
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
            exclude_modules = [
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.add_q_proj",
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.add_k_proj",
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.add_v_proj",
                f"transformer_blocks.{config_model.num_transformer_layers - 1}.attn.to_add_out",
            ]
            if not config_model.use_text_features or config_model.text_head_idx[-1] != config_model.num_transformer_layers:
                exclude_modules = None

            lora_config = LoraConfig(
                r=config_model.lora_config.r,
                lora_alpha=config_model.lora_config.lora_alpha,
                init_lora_weights=config_model.lora_config.init_lora_weights,
                target_modules=target_modules,
                exclude_modules=exclude_modules,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            # self.backbone.to(dtype=dtype)

        backbone_dim = pipeline.transformer.inner_dim
        self.backbone_dim = backbone_dim
        self.reward_head = RewardHeadV3(
            token_dim=backbone_dim,
            n_visual_heads=len(config_model.visual_head_idx),
            n_text_heads=len(config_model.text_head_idx),
            **getattr(config_model, "reward_head", {})
        )

        # self.reward_head = self.reward_head.to(dtype=dtype)

    def encode_prompt(self, prompts):
        with ms._no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                self.text_encoders, self.tokenizers, prompts, max_sequence_length=128
            )
        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds
        }

    def construct(
        self,
        latents: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        pooled_projections: Optional[ms.Tensor],
        timesteps: ms.Tensor,
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
            hw=(h, w)
        )

        return reward


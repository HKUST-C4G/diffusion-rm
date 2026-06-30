"""Diffusion-based reward model (FLUX)."""

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
    max_sequence_length=512,
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
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="np",
        )
        text_input_ids = ms.Tensor(np.asarray(text_inputs.input_ids), dtype=ms.int32)
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids)[0]
    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
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
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="np",
        )

        text_input_ids = ms.Tensor(np.asarray(text_inputs.input_ids), dtype=ms.int32)
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)
    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype

    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
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
    )

    text_ids = mint.zeros(prompt_embeds.shape[1], 3)

    return prompt_embeds, pooled_prompt_embeds, text_ids


class FLUXBackbone(nn.Cell):
    def __init__(self, transformer, config_model):
        super().__init__()
        self.pos_embed = transformer.pos_embed
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder
        self.x_embedder = transformer.x_embedder

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
        img_ids: ms.Tensor = None,
        txt_ids: ms.Tensor = None,
        guidance: ms.Tensor = None,
    ) -> ms.Tensor:
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = mint.cat((txt_ids, img_ids), dim=0)

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


class FLUXRewardModel(nn.Cell):
    """Diffusion-based reward model using pretrained transformer backbone."""

    def __init__(self, pipeline, config_model, vae_scale_factor):
        super().__init__()
        ## NOTE: All the modules should be moved to the target device and dtype before here!!!
        self.config_model = config_model
        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2

        text_encoder_1.requires_grad = False
        text_encoder_2.requires_grad = False

        self.text_encoders = [text_encoder_1, text_encoder_2]
        self.tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

        self.backbone = FLUXBackbone(
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
                "attn2.to_q",
                "attn2.to_k",
                "attn2.to_v",
                "attn2.to_out.0",
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

        backbone_dim = pipeline.transformer.inner_dim
        self.backbone_dim = backbone_dim
        self.reward_head = RewardHeadV3(
            token_dim=backbone_dim,
            n_visual_heads=len(config_model.visual_head_idx),
            n_text_heads=len(config_model.text_head_idx),
            **getattr(config_model, "reward_head", {})
        )

        self.vae_scale_factor = vae_scale_factor

    def encode_prompt(self, prompts):
        with ms._no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                self.text_encoders, self.tokenizers, prompts, max_sequence_length=128
            )

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
    def _prepare_latent_image_ids(batch_size, height, width):
        latent_image_ids = mint.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + mint.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + mint.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids

    def construct(
        self,
        latents: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        pooled_projections: Optional[ms.Tensor],
        txt_ids: ms.Tensor,
        timesteps: ms.Tensor,
    ) -> ms.Tensor:
        b, c, h, w = latents.shape
        latents = self._pack_latents(latents, b, c, h, w)

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size=b,
            height=h // 2,
            width=w // 2,
        )

        guidance = ms.Tensor([3.5])

        temb, hidden_states_list, encoder_hidden_states_list = self.backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps,
            img_ids=latent_image_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        reward = self.reward_head(
            visual_features=hidden_states_list,
            text_features=encoder_hidden_states_list,
            t_embed=temb,
            hw=(h, w)
        )

        return reward


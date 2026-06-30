"""
Diffusion-RM 推理接口

加载 Diffusion-RM 的预训练权重，在推理时计算 reward。
不涉及训练逻辑，仅提供 scorer 接口。
"""

import os
from typing import Dict, List, Optional, Union

import mindspore as ms
from mindspore import mint
import numpy as np
from PIL import Image

from .scorer import Scorer
from .diffusion_rm_impl import DRMInferencer


class DiffusionRMFluxScorer(Scorer):
    """
    FLUX-based Diffusion Reward Model Scorer
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        pipeline=None,
        pipeline_path: str = "black-forest-labs/FLUX.1-dev",
        u: float = 0.9,
    ):
        super().__init__()

        self.u = u

        if pipeline is None:
            print(f"Loading FLUX pipeline from {pipeline_path}...")
            from diffusers import FluxPipeline

            self.pipeline = FluxPipeline.from_pretrained(
                pipeline_path,
                dtype=ms.bfloat16,
            )
        else:
            self.pipeline = pipeline

        print(f"Loading Diffusion-RM from {checkpoint_path}...")
        self.drm = DRMInferencer(
            pipeline=self.pipeline,
            config_path=config_path,
            model_path=checkpoint_path,
            model_dtype=ms.bfloat16,
        )

        print("Diffusion-RM loaded successfully!")

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        pil_images = self._to_pil_images(images)
        latents = self._encode_images_to_latents(pil_images)

        if prompts is None:
            prompts = [""] * len(pil_images)
        text_conds = self._encode_prompts(prompts)

        rewards = self.drm.reward(
            text_conds=text_conds,
            latents=latents,
            u=self.u,
        )

        scores = (rewards.float() / 5.0).reshape((-1,)).asnumpy().tolist()
        return [float(s) for s in scores]

    def reward(
        self,
        latents: ms.Tensor,
        text_conds: Dict[str, ms.Tensor],
        u: Optional[float] = None,
    ) -> ms.Tensor:
        if u is None:
            u = self.u

        latents_pt = ms.Tensor(latents.asnumpy())
        text_conds_pt = {
            k: ms.Tensor(v.asnumpy())
            for k, v in text_conds.items()
        }

        rewards_pt = self.drm.reward(
            text_conds=text_conds_pt,
            latents=latents_pt,
            u=u,
        )

        return ms.Tensor(rewards_pt.asnumpy())

    def _to_pil_images(self, images: Union[List[Image.Image], np.ndarray, ms.Tensor]) -> List[Image.Image]:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            return images
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [Image.fromarray(images)]
            else:
                images = [Image.fromarray(img) for img in images]
        elif isinstance(images, ms.Tensor):
            np_images = images.asnumpy()
            if np_images.ndim == 3:
                np_images = np_images.transpose(1, 2, 0)
                images = [Image.fromarray(np_images)]
            else:
                np_images = np_images.transpose(0, 2, 3, 1)
                images = [Image.fromarray(img) for img in np_images]
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        return images

    def _vae_encode_to_latents(self, pixel_values: ms.Tensor) -> ms.Tensor:
        enc = self.pipeline.vae.encode(pixel_values)[0]
        return self.pipeline.vae.diag_gauss_dist.mode(enc)

    def _encode_images_to_latents(self, images: List[Image.Image]) -> ms.Tensor:
        #self.pipeline.vae.eval()

        latents_list = []
        with ms._no_grad():
            for img in images:
                img = img.convert("RGB")
                pixel_values = self.pipeline.image_processor.preprocess(img).to(
                    dtype=ms.float32
                )

                latents = self._vae_encode_to_latents(pixel_values)
                # Match diffusers latent convention when applicable
                scaling_factor = getattr(getattr(self.pipeline.vae, "config", None), "scaling_factor", None)
                shift_factor = getattr(getattr(self.pipeline.vae, "config", None), "shift_factor", None)
                if scaling_factor is not None and shift_factor is not None:
                    latents = (latents - shift_factor) * scaling_factor
                elif scaling_factor is not None:
                    latents = latents * scaling_factor
                latents_list.append(latents)

        latents = mint.cat(latents_list, dim=0)
        return latents

    def _encode_prompts(self, prompts: List[str]) -> Dict[str, ms.Tensor]:
        from .diffusion_rm_impl.models.flux_rm import encode_prompt

        with ms.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders=[
                    self.pipeline.text_encoder,
                    self.pipeline.text_encoder_2,
                ],
                tokenizers=[
                    self.pipeline.tokenizer,
                    self.pipeline.tokenizer_2,
                ],
                prompt=prompts,
                max_sequence_length=128,
            )

            prompt_embeds = prompt_embeds.to(dtype=ms.float32)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=ms.float32)
            text_ids = text_ids.to(dtype=ms.float32)

        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
        }


class DiffusionRMSD3Scorer(Scorer):
    """
    SD3-based Diffusion Reward Model Scorer
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        pipeline=None,
        pipeline_path: str = "stabilityai/stable-diffusion-3.5-medium",
        u: float = 0.9,
    ):
        super().__init__()

        self.u = u

        if pipeline is None:
            print(f"Loading SD3 pipeline from {pipeline_path}...")
            from mindone.diffusers import StableDiffusion3Pipeline

            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                pipeline_path,
                ms_dtype=ms.bfloat16,
            )
            # VAE encode is more stable in fp32
            if hasattr(self.pipeline, "vae"):
                self.pipeline.vae.to(dtype=ms.float32)
        else:
            self.pipeline = pipeline

        print(f"Loading Diffusion-RM from {checkpoint_path}...")
        self.drm = DRMInferencer(
            pipeline=self.pipeline,
            config_path=config_path,
            model_path=checkpoint_path,
            model_dtype=ms.bfloat16,
        )

        print("Diffusion-RM loaded successfully!")

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        pil_images = self._to_pil_images(images)
        latents = self._encode_images_to_latents(pil_images)

        if prompts is None:
            prompts = [""] * len(pil_images)
        text_conds = self._encode_prompts(prompts)

        rewards = self.drm.reward(
            text_conds=text_conds,
            latents=latents,
            u=self.u,
        )

        scores = (rewards.float() / 5.0).reshape((-1,)).asnumpy().tolist()
        return [float(s) for s in scores]

    def reward(
        self,
        latents: ms.Tensor,
        text_conds: Dict[str, ms.Tensor],
        u: Optional[float] = None,
    ) -> ms.Tensor:
        if u is None:
            u = self.u

        rewards = self.drm.reward(
            text_conds=text_conds,
            latents=latents,
            u=u,
        )

        return rewards

    def _to_pil_images(self, images: Union[List[Image.Image], np.ndarray, ms.Tensor]) -> List[Image.Image]:
        if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            return images
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [Image.fromarray(images)]
            else:
                images = [Image.fromarray(img) for img in images]
        elif isinstance(images, ms.Tensor):
            np_images = images.asnumpy()
            if np_images.ndim == 3:
                np_images = np_images.transpose(1, 2, 0)
                images = [Image.fromarray(np_images)]
            else:
                np_images = np_images.transpose(0, 2, 3, 1)
                images = [Image.fromarray(img) for img in np_images]
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")

        return images

    def _encode_images_to_latents(self, images: List[Image.Image]) -> ms.Tensor:
        #self.pipeline.vae.eval()

        latents_list = []
        with ms._no_grad():
            for img in images:
                img = img.convert("RGB")
                pixel_values = self.pipeline.image_processor.preprocess(img).to(
                    dtype=ms.float32
                )

                # reuse the same moments->latents conversion as FLUX scorer
                latents = DiffusionRMFluxScorer._vae_encode_to_latents(self, pixel_values)
                scaling_factor = getattr(getattr(self.pipeline.vae, "config", None), "scaling_factor", None)
                shift_factor = getattr(getattr(self.pipeline.vae, "config", None), "shift_factor", None)
                if scaling_factor is not None and shift_factor is not None:
                    latents = (latents - shift_factor) * scaling_factor
                elif scaling_factor is not None:
                    latents = latents * scaling_factor
                latents_list.append(latents)

        latents = mint.cat(latents_list, dim=0)
        return latents

    def _encode_prompts(self, prompts: List[str]) -> Dict[str, ms.Tensor]:
        from .diffusion_rm_impl.models.sd3_rm import encode_prompt

        with ms._no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=[
                    self.pipeline.text_encoder,
                    self.pipeline.text_encoder_2,
                    self.pipeline.text_encoder_3,
                ],
                tokenizers=[
                    self.pipeline.tokenizer,
                    self.pipeline.tokenizer_2,
                    self.pipeline.tokenizer_3,
                ],
                prompt=prompts,
                max_sequence_length=128,
            )

            prompt_embeds = prompt_embeds.to(dtype=ms.float32)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=ms.float32)

        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
        }

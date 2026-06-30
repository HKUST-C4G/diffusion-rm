from typing import Optional, Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindone.diffusers import SD3Transformer2DModel

from .pipelines import StableDiffusion3PipelineWithSDELogProb
from .schedulers import FlowMatchEulerSDEDiscreteScheduler
from .utils import compute_log_prob


class NetWithLoss(nn.Cell):

    def __init__(
        self,
        pipeline: StableDiffusion3PipelineWithSDELogProb,
        guidance_scale: float = 1.0,
        adv_clip_max: float = 5.0,
        beta: float = 0.001,
        clip_range: float = 1e-4,
    ) -> None:
        super().__init__()
        self.transformer: SD3Transformer2DModel = pipeline.transformer
        self.scheduler: FlowMatchEulerSDEDiscreteScheduler = pipeline.scheduler
        self.guidance_scale = guidance_scale
        self.adv_clip_max = adv_clip_max
        self.beta = beta
        self.clip_range = clip_range

    def compute_log_prob(
        self,
        latents: ms.Tensor,
        next_latents: ms.Tensor,
        timesteps: ms.Tensor,
        embeds: ms.Tensor,
        pooled_embeds: ms.Tensor,
        sigma: ms.Tensor,
        sigma_next: ms.Tensor,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        if self.guidance_scale > 1.0:
            noise_pred = self.transformer(
                hidden_states=mint.cat([latents] * 2),
                timestep=mint.cat([timesteps] * 2),
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        # compute the log prob of next_latents given latents under the current model
        prev_sample_mean, var_t, dt = self.scheduler.compute_prev_sample_mean(
            noise_pred, latents.float(), sigma, sigma_next
        )

        log_prob = compute_log_prob(next_latents, prev_sample_mean, var_t, dt)

        return log_prob, prev_sample_mean, -var_t * dt

    def construct(
        self,
        latents: ms.Tensor,
        next_latents: ms.Tensor,
        timesteps: ms.Tensor,
        embeds: ms.Tensor,
        pooled_embeds: ms.Tensor,
        advantages: ms.Tensor,
        sample_log_probs: ms.Tensor,
        sigma: ms.Tensor,
        sigma_next: ms.Tensor,
        prev_sample_mean_ref: Optional[ms.Tensor] = None,
        loss_scaler: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        if self.beta > 0:
            assert prev_sample_mean_ref is not None

        log_prob, prev_sample_mean, var_t = self.compute_log_prob(
            latents, next_latents, timesteps, embeds, pooled_embeds, sigma, sigma_next
        )
        # grpo logic
        advantages = mint.clamp(
            advantages,
            -self.adv_clip_max,
            self.adv_clip_max,
        )
        ratio = mint.exp(log_prob - sample_log_probs)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * mint.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        policy_loss = mint.mean(mint.maximum(unclipped_loss, clipped_loss))
        if self.beta > 0:
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
                dim=(1, 2, 3), keepdim=True
            ) / (2 * var_t)
            kl_loss = mint.mean(kl_loss)
            loss = policy_loss + self.beta * kl_loss
        else:
            loss = policy_loss

        if loss_scaler is not None:
            loss = loss * loss_scaler

        return loss

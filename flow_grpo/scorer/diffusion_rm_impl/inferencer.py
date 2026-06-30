import os
from typing import Optional

import mindspore as ms
from mindspore import mint
from .config_utils import load_config
from .models.sd3_rm import SD3RewardModel


def get_timesteps_from_u(noise_scheduler, u: ms.Tensor, n_dim: int = 4):
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices]

    sigmas = noise_scheduler.sigmas[indices]
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas[:, None]
    return sigmas, timesteps


class DRMInferencer:
    """Self-contained inference engine for Diffusion-RM checkpoints."""

    def __init__(
        self,
        pipeline,
        config_path: str,
        model_path: str,
        model_dtype: ms.dtype = ms.bfloat16,
    ):
        self.config = load_config(config_path)
        self.model_dtype = model_dtype


        backbone_id = ""
        try:
            backbone_id = str(self.config.model.backbone_model_id).lower()
        except Exception:
            backbone_id = ""
        pipe_cls = type(pipeline).__name__.lower()
        is_flux = ("flux" in backbone_id) or ("flux" in pipe_cls)

        if is_flux:
            from .models.flux_rm import FLUXRewardModel

            vae_scale_factor = getattr(pipeline, "vae_scale_factor", None)
            if vae_scale_factor is None:
                vae_scale_factor = getattr(getattr(pipeline, "vae", None), "vae_scale_factor", 8)
            self.model = FLUXRewardModel(
                pipeline=pipeline,
                config_model=self.config.model,
                vae_scale_factor=vae_scale_factor,
            )
        else:
            self.model = SD3RewardModel(
                pipeline=pipeline,
                config_model=self.config.model,
                dtype=model_dtype,
            )

        scheduler_cls = type(pipeline.scheduler)
        self.noise_scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

        self.add_noise = True
        try:
            if "training" in self.config and "add_noise" in self.config.training:
                self.add_noise = bool(self.config.training.add_noise)
        except Exception:
            self.add_noise = True

        self.load_checkpoint(model_path)

    def reward(self, text_conds, latents: ms.Tensor, u: float = 0.9) -> ms.Tensor:
        self.model.set_train(False)
        ori_adapter = None
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "set_adapter"):
            ori_adapter = getattr(backbone, "active_adapter", None)
            available = []
            try:
                peft_cfg = getattr(backbone, "peft_config", None)
                if isinstance(peft_cfg, dict):
                    available = list(peft_cfg.keys())
            except Exception:
                available = []

            if (not available) or ("default" in available):
                backbone.set_adapter("default")


        bsz = latents.shape[0]

        u_tensor = ms.tensor([u] * bsz)
        sigmas, timesteps = get_timesteps_from_u(self.noise_scheduler, u_tensor, n_dim=len(latents.shape))

        noise = mint.randn_like(latents)
        if self.add_noise:
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        else:
            noisy_model_input = latents

        noisy_model_input = noisy_model_input.to(self.model_dtype)

        casted = {}
        for k, v in text_conds.items():
            casted[k] = v.to(self.model_dtype)
        text_conds = casted

        score = self.model(
            latents=noisy_model_input,
            timesteps=timesteps,
            **text_conds,
        )

        if ori_adapter is not None and backbone is not None and hasattr(backbone, "set_adapter"):
            try:
                backbone.set_adapter(ori_adapter)
            except Exception:
                pass

        return score

    def load_checkpoint(self, checkpoint_path: str):
        def _load_ms_ckpt_into_net(net: ms.nn.Cell, ckpt_path: str):
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"MindSpore checkpoint not found: {ckpt_path}. "
                    f"Please convert Diffusion-RM weights to .ckpt first."
                )

            ckpt = ms.load_checkpoint(ckpt_path)
            ms.load_param_into_net(net, ckpt, strict_load=False)

        if getattr(self.config.model, "use_lora", False):
            # 1) backbone LoRA weights
            lora_ckpt = os.path.join(checkpoint_path, "backbone_lora", "adapter_model.ckpt")
            if os.path.exists(lora_ckpt):
                _load_ms_ckpt_into_net(self.model.backbone, lora_ckpt)
            else:
                print(f"Warning: LoRA checkpoint not found, using base backbone: {lora_ckpt}")

            # 2) reward head
            rm_head_ckpt = os.path.join(checkpoint_path, "rm_head.ckpt")
            _load_ms_ckpt_into_net(self.model.reward_head, rm_head_ckpt)


        elif not getattr(self.config.model, "freeze_backbone", False):
            full_model_ckpt = os.path.join(checkpoint_path, "full_model.ckpt")
            _load_ms_ckpt_into_net(self.model, full_model_ckpt)

        else:
            rm_head_ckpt = os.path.join(checkpoint_path, "rm_head.ckpt")
            _load_ms_ckpt_into_net(self.model.reward_head, rm_head_ckpt)

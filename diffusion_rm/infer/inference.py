from sklearn import pipeline
import yaml
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from typing import Dict, Any
import os

from diffusion_rm.models.sd3_rm import SD3RewardModel
from huggingface_hub import snapshot_download

import logging
logger = logging.getLogger(__name__)


def get_timesteps_from_u(noise_scheduler, u: torch.Tensor, n_dim: int=4, dtype: torch.dtype=torch.float32):
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps.to(u.device)[indices]

    # get sigmas
    sigmas = noise_scheduler.sigmas.to(u.device)[indices]

    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas, timesteps

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return OmegaConf.create(config)

class DRMInferencer:
    """Inference engine for the diffusion reward model."""
    def __init__(
        self,
        pipeline,
        config_path,
        model_path,
        load_from_disk=False,
        device='cuda',
        model_dtype=torch.bfloat16,
    ):
        print(f"==========device: {device}===============")

        if load_from_disk:
            logger.info(f"Loading config from disk: {config_path}")
        else:
            logger.info(f"Downloading weights from HF: {model_path}")
            local_model_path = snapshot_download(
                repo_id=model_path,
            )
            checkpoint_path = os.path.join(local_model_path, "checkpoint")
            config_path = os.path.join(local_model_path, "config.json")

        config = load_config(config_path)
    
        sd3_rm = SD3RewardModel(
            pipeline=pipeline,
            config_model=config.model,
            device=device,
            dtype=model_dtype,
        )
        self.model = sd3_rm

        scheduler_cls = type(pipeline.scheduler)
        self.noise_scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
        self.config = config

        self.add_noise = config.training.add_noise if 'training' in config and 'add_noise' in config.training else True
        # self.add_noise = False

        if load_from_disk:        
            logger.info(f"Loading config from disk: {model_path}")
            self.load_checkpoint(model_path, device)
        else:
            self.load_checkpoint(checkpoint_path, device)

        self.model_dtype = model_dtype


    @staticmethod
    def get_timesteps_from_u(noise_scheduler, u: torch.Tensor, n_dim: int=4, dtype: torch.dtype=torch.float32):
        # raise NotImplementedError("This method should be implemented in train script")
        return get_timesteps_from_u(noise_scheduler, u, n_dim, dtype)

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

    def reward(self, text_conds, latents, u=0.1) -> Dict[str, float]:
        self.model.eval()
        ori_adapter = self.model.backbone.active_adapter
        self.model.backbone.set_adapter('rm_lora')

        device = latents.device

        # latents = latents.to(self.model_dtype)
        # text_conds = {k: v.to(self.model_dtype) for k, v in text_conds.items()}
                
        # Random add noise
        bsz = latents.shape[0]

        u_tensor = torch.tensor([u] * bsz, device=device)
        sigmas, timesteps = self.get_timesteps_from_sigma(self.noise_scheduler, u_tensor, n_dim=len(latents.shape), dtype=latents.dtype)

        noise = torch.randn_like(latents)

        if self.add_noise:
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        else:
            noisy_model_input = latents

        noisy_model_input = noisy_model_input.to(self.model_dtype)
        # import pdb; pdb.set_trace()
        score = self.model(
            latents=noisy_model_input,
            timesteps=timesteps,
            **text_conds,
        )

        self.model.backbone.set_adapter(ori_adapter)

        return score

    def load_checkpoint(self, checkpoint_path: str, device='cuda'):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if self.config.model.use_lora:
            # Load LoRA weights
            lora_dir = os.path.join(checkpoint_path, "backbone_lora")
            if os.path.exists(lora_dir):
                logger.info(f"Loading LoRA weights from: {lora_dir}")
                self.model.backbone.load_adapter(lora_dir, adapter_name="rm_lora")
            else:
                logger.info(f"Warning: LoRA directory not found: {lora_dir}")
                
            # Load reward head
            rm_head_path = os.path.join(checkpoint_path, "rm_head.pt")
            if os.path.exists(rm_head_path):
                logger.info(f"Loading reward head from: {rm_head_path}")
                reward_head_state = torch.load(rm_head_path, map_location=device)
                self.model.reward_head.load_state_dict(reward_head_state)
            else:
                logger.info(f"Warning: Reward head file not found: {rm_head_path}")
                
        elif not self.config.model.freeze_backbone:
            # Load full model
            full_model_path = os.path.join(checkpoint_path, "full_model.pt")
            if os.path.exists(full_model_path):
                logger.info(f"Loading full model from: {full_model_path}")
                model_state = torch.load(full_model_path, map_location=device)
                self.model.load_state_dict(model_state)
            else:
                logger.info(f"Warning: Full model file not found: {full_model_path}")
        
        else:
            # Load only reward head (backbone is frozen)
            rm_head_path = os.path.join(checkpoint_path, "rm_head.pt")
            if os.path.exists(rm_head_path):
                logger.info(f"Loading reward head from: {rm_head_path}")
                reward_head_state = torch.load(rm_head_path, map_location=device)
                self.model.reward_head.load_state_dict(reward_head_state)
            else:
                logger.info(f"Warning: Reward head file not found: {rm_head_path}")

    

def test_reward_from_generated():
    from diffusers import StableDiffusion3Pipeline
    from diffusion_rm.models.sd3_rm import encode_prompt
    # from fvcore.nn import FlopCountAnalysis

    
    ## Step 1: load the SD3.5-M pipeline and the lora scorer
    device = torch.device('cuda:0')
    dtype=torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        dtype=dtype,
        device=device
    )
    pipe.vae.to(device, dtype=dtype)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.text_encoder_2.to(device, dtype=dtype)
    pipe.text_encoder_3.to(device, dtype=dtype)

    pipe.transformer.to(device, dtype=dtype)

    scorer = DRMInferencer(
        pipeline=pipe,
        config_path=None,
        model_path="liuhuohuo/DiNa-LRM-SD35M-12layers",
        device=device,
        model_dtype=dtype,
        load_from_disk=False,
    )


    ## 2. generate latents;
    # encode prompt
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    def compute_text_embeddings(text_encoders, tokenizers, prompts):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompts, max_sequence_length=256
            )
            prompt_embeds = prompt_embeds.to(text_encoders[0].device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(text_encoders[0].device)

        return prompt_embeds, pooled_prompt_embeds
    
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        text_encoders, 
        tokenizers,
        ["A girl walking in the street"], 
    )
    
    output = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=40,
        guidance_scale=4.5,
        output_type='latent'
    )
    latents = output.images


    ## 3. compute reward
    print("Computing rewards")
    with torch.no_grad():
        raw_score = scorer.reward(
            text_conds={
                'encoder_hidden_states': prompt_embeds,
                'pooled_projections': pooled_prompt_embeds
            },
            latents=latents,
            u = 0.4,
        )
        score = (raw_score + 10.0) / 10.0
        print(f"reward value: {raw_score}")

    ## 4. [Optional] decode and save images
    with torch.no_grad():
        latents_decoded = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents_decoded.to(pipe.vae.dtype), return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        
    image.save("example.png")


def test_reward_from_disk():
    from diffusers import StableDiffusion3Pipeline
    from diffusion_rm.models.sd3_rm import encode_prompt
    from PIL import Image
    import torchvision.transforms as T
    # from fvcore.nn import FlopCountAnalysis

    
    ## Step 1: load the SD3.5-M pipeline and the lora scorer
    device = torch.device('cuda:0')
    dtype=torch.bfloat16
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        dtype=dtype,
        device=device
    )
    pipe.vae.to(device, dtype=dtype)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.text_encoder_2.to(device, dtype=dtype)
    pipe.text_encoder_3.to(device, dtype=dtype)

    pipe.transformer.to(device, dtype=dtype)

    scorer = DRMInferencer(
        pipeline=pipe,
        config_path=None,
        model_path="liuhuohuo/DiNa-LRM-SD35M-12layers",
        device=device,
        model_dtype=dtype,
        load_from_disk=False,
    )


    ## 2. load image, and encode to latent
    # encode prompt
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    def compute_text_embeddings(text_encoders, tokenizers, prompts):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompts, max_sequence_length=256
            )
            prompt_embeds = prompt_embeds.to(text_encoders[0].device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(text_encoders[0].device)

        return prompt_embeds, pooled_prompt_embeds
    
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        text_encoders, 
        tokenizers,
        ["A girl walking in the street"], 
    )
    

    ## load image file
    image_path = "assets/example.png"
    print(f"Processing local image: {image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(raw_image).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


    ## 3. compute reward
    print("Computing rewards")
    with torch.no_grad():
        raw_score = scorer.reward(
            text_conds={
                'encoder_hidden_states': prompt_embeds,
                'pooled_projections': pooled_prompt_embeds
            },
            latents=latents,
            u = 0.4,
        )
        score = (raw_score + 10.0) / 10.0
        print(f"reward value: {raw_score}")
    

if __name__ == "__main__":
    test_reward_from_generated()


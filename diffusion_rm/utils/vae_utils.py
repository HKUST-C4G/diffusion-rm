"""VAE utilities for encoding/decoding images."""

import torch



class VAEProcessor:
    def __init__(self, vae):
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae_config_shift_factor = vae.config.shift_factor
        self.vae_config_scaling_factor = vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images):
        latents = self.vae.encode(images.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample()
        latents = (latents - self.vae_config_shift_factor) * self.vae_config_scaling_factor
        return latents
        
    @torch.no_grad()
    def decode(self, latents):
        latents = latents / self.vae_config_scaling_factor + self.vae_config_shift_factor
        images = self.vae.decode(latents.to(self.vae.device, dtype=self.vae.dtype)).sample
        return images

class VAEProcessor_QwenImage:
    def __init__(self, vae):
        self.vae = vae
        self.vae.requires_grad_(False)

        self.vae_config_latents_mean = torch.tensor(vae.config.latents_mean).to(vae.device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
        self.vae_config_latents_std = torch.tensor(vae.config.latents_std).to(vae.device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)

    @torch.no_grad()
    def encode(self, images):
        pixel_values = images.to(self.vae.device, dtype=self.vae.dtype)
        # Qwen expects a `num_frames` dimension too.
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - self.vae_config_latents_mean) / self.vae_config_latents_std
        return latents
        
    @torch.no_grad()
    def decode(self, latents):
        raise NotImplementedError("Decoding not implemented for VAEProcessor_QwenImage")
        latents = latents * self.vae_config_latents_std + self.vae_config_latents_mean
        images = self.vae.decode(latents.to(self.vae.device, dtype=self.vae.dtype)).sample
        return images
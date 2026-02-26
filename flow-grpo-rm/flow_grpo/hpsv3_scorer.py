from email.mime import image
from re import S
from PIL import Image
from typing import Optional, Union
import torch
import numpy as np

from hpsv3 import HPSv3RewardInferencer
from hpsv3.dataset.utils import fetch_image
from hpsv3.dataset.data_collator_qwen import prompt_with_special_token, prompt_without_special_token, INSTRUCTION

from torchvision import transforms

import torch
import torch.nn.functional as F
from transformers.image_utils import (
    ChannelDimension,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    to_numpy_array,
    make_list_of_images,
    is_valid_image,
    valid_images,
    validate_preprocess_arguments
)

from transformers.utils import logging
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import convert_to_rgb, to_channel_dimension_format
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs

logger = logging.get_logger(__name__)


def process_image_tensor(
    images,
):
    images = make_list_of_images(images)

    if isinstance(images[0], torch.Tensor):
        patch_size = 14 # hard code here
        temporal_patch_size = 2
        merge_size = 2

        # 按原逻辑做 patch 展开
        patches = torch.stack(images, dim=0)  # [N, C, H, W]

        if patches.shape[0] == 1:
            patches = patches.repeat_interleave(temporal_patch_size, dim=0).contiguous()

        height, width = patches.shape[2:]
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = height // patch_size, width // patch_size

        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        ).contiguous()
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        ).contiguous()

        return flatten_patches, (grid_t, grid_h, grid_w)

    raise ValueError("Only torch.Tensor input is supported in this custom preprocess function.")


def process(
    self,
    images=None,
    text=None,
    **kwargs,
) -> BatchFeature:
    output_kwargs = self._merge_kwargs(
        Qwen2VLProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )
    if images is not None:
        pixel_values, vision_grid_thws = [], []
        for image in images:
            image_tensor, image_grid_thw = process_image_tensor(images=image)
            pixel_values.append(image_tensor)
            vision_grid_thws.append(image_grid_thw)
            vision_grid_thws = torch.tensor(vision_grid_thws)

        image_inputs = {
            "pixel_values": torch.stack(pixel_values),
            "image_grid_thw": vision_grid_thws,
        }
        image_grid_thw = image_inputs["image_grid_thw"]

    if not isinstance(text, list):
        text = [text]

    if image_grid_thw is not None:
        merge_length = self.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|image_pad|>" in text[i]:
                text[i] = text[i].replace(
                    "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

    _ = output_kwargs["text_kwargs"].pop("padding_side", None)
    text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

    return BatchFeature(data={**text_inputs, **image_inputs})

class HPSv3Rewarder(HPSv3RewardInferencer):
    def prepare_batch(self, images, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        image_inputs = []
        for text, image in zip(prompts, images):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": max_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)
            image_inputs.append(fetch_image({"image": image}))

        batch = process(
            self.processor,
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": False, "do_normalize": True},    # not rescale for tensor
        )
        batch = self._prepare_inputs(batch)
        return batch

    # @torch.inference_mode()
    def reward(self, prompts, image_paths):
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        return rewards
        # class RewardModelWrapper(torch.nn.Module):
        #     def __init__(self, model):
        #         super().__init__()
        #         self.model = model

        #     def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        #         out = self.model(
        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             pixel_values=pixel_values,
        #             image_grid_thw=image_grid_thw,
        #             return_dict=True,
        #         )
        #         return out["logits"]

        # wrapper = RewardModelWrapper(self.model)
        # from fvcore.nn import FlopCountAnalysis
        # inputs = (batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_grid_thw"])
        # flops_ana = FlopCountAnalysis(wrapper, inputs)
        # total_flops = flops_ana.total()
        # print("Total FLOPs:", total_flops, "≈", total_flops / 1e12, "TFLOPs")

class HPSv3Scorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = HPSv3Rewarder(device=device)

        def _transform():
            return transforms.Compose(
                [
                    transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        self.transform = _transform()
        
    # @torch.no_grad()
    def __call__(self, prompts, images):
        rewards = []
        for prompt,image in zip(prompts, images):
            image = self.transform(image)
            reward = self.model.reward(prompts=[prompt], image_paths=image.unsqueeze(0))
            rewards.append(reward[:, 0])

        rewards = torch.cat(rewards, dim=0)
        rewards = rewards / 10.0
        return rewards

# Usage example
def main():
    scorer = HPSv3Scorer(
        device="cuda",
        dtype=torch.float32
    )

    images=[
    "/m2v_intern/liugongye/code/00_update/flow_grpo/scripts/demo/cat.png",
    ]
    pil_images = [Image.open(img).convert("RGB") for img in images]
    # to tensor [B, C, H, W]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensors = [transform(img).to(device="cuda") for img in pil_images]
    image_tensors = torch.cat(image_tensors, dim=0).to(device=scorer.device, dtype=scorer.dtype).unsqueeze(0)

    prompts=[
        'A cat sitting on a windowsill',
    ]
    print(scorer(prompts, image_tensors))

if __name__ == "__main__":
    main()
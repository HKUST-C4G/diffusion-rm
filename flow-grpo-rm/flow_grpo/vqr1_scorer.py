import re
from typing import List, Union, Optional

import torch
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class VisualQualityR1Scorer(torch.nn.Module):
    """
    VisualQuality-R1-7B scorer
    - Input: prompts (List[str]), images (List[PIL] or torch.Tensor in [-1,1])
    - Output: scores (Tensor[B]) normalized to [0,1] by default (divide by 5).
    """
    def __init__(
        self,
        model_path: str = "TianheWu/VisualQuality-R1-7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        batch_size: int = 8,
        normalize: bool = True,      # return score/5
        use_thinking: bool = False,  # False = non-thinking (recommended for reward/eval)
        max_new_tokens: int = 32,    # enough for "<answer>4.12</answer>"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        self.normalize = normalize
        self.use_thinking = use_thinking
        self.max_new_tokens = max_new_tokens

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).eval().to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_path)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        self.PROMPT = (
            "You are doing the image quality assessment task. Here is the question: "
            "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
            "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
        )
        if self.use_thinking:
            self.QUESTION_TEMPLATE = (
                "{Question} First output the thinking process in <think> </think> tags and then output "
                "the final answer with only one score in <answer> </answer> tags."
            )
        else:
            self.QUESTION_TEMPLATE = (
                "{Question} Please only output the final answer with only one score in <answer> </answer> tags."
            )

    @staticmethod
    def _tensor_to_pil(img_chw: torch.Tensor) -> Image.Image:
        """
        Accept tensor [C,H,W] in [-1,1] (preferred) or [0,1] or [0,255].
        Convert to PIL RGB.
        """
        if img_chw.ndim != 3:
            raise ValueError(f"Expect [C,H,W], got {tuple(img_chw.shape)}")

        x = img_chw.detach().to(torch.float32).cpu()  # [C,H,W]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] >= 3:
            x = x[:3]  # drop alpha or extra channels
        else:
            raise ValueError(f"Unsupported channel count C={x.shape[0]}")

        mn = float(x.min())
        mx = float(x.max())

        # Range handling
        if mn >= -1.01 and mx <= 1.01:
            # either [-1,1] or [0,1]
            if mn < 0.0:
                x = (x + 1.0) * 0.5  # [-1,1] -> [0,1]
            # else already [0,1]
            x = x.clamp(0.0, 1.0) * 255.0
        else:
            # likely [0,255] already (float) or something close
            # (we still clamp to be safe)
            x = x.clamp(0.0, 255.0)

        x = x.round().to(torch.uint8)  # [C,H,W] uint8
        x = x.permute(1, 2, 0).contiguous()  # HWC
        return Image.fromarray(x.numpy(), mode="RGB")

    def _coerce_images(self, images: Union[List[Image.Image], torch.Tensor]) -> List[Image.Image]:
        """
        Simplified assumption for tensor:
          - only accept [B,C,H,W] or [C,H,W]
          - channel C in {1,3,4}
          - value range typically [-1,1]
        """
        if isinstance(images, list):
            pil_images = []
            for im in images:
                if not isinstance(im, Image.Image):
                    raise TypeError(f"List must contain PIL.Image, got {type(im)}")
                pil_images.append(im.convert("RGB"))
            return pil_images

        if torch.is_tensor(images):
            x = images
            if x.ndim == 3:
                x = x.unsqueeze(0)  # [1,C,H,W]
            if x.ndim != 4:
                raise ValueError(f"Tensor images must be [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
            if x.shape[1] not in (1, 3, 4):
                raise ValueError(f"Expect C in (1,3,4), got C={x.shape[1]}")

            return [self._tensor_to_pil(x[i]) for i in range(x.shape[0])]

        raise TypeError(f"Unsupported images type: {type(images)}")

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        """
        Prefer <answer>...</answer>, fallback to first float found.
        """
        try:
            matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
            candidate = matches[-1].strip() if matches else text.strip()
            m = re.search(r"\d+(\.\d+)?", candidate)
            if m is None:
                return None
            score = float(m.group())
            return float(min(5.0, max(1.0, score)))
        except Exception:
            return None

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str] = None,
        return_raw: bool = False,
    ) -> torch.Tensor:
        pil_images = self._coerce_images(images)
        if not isinstance(prompts, list):
            raise TypeError("prompts must be a List[str]")
        assert len(prompts) == len(pil_images), f"len(prompts)={len(prompts)} != len(images)={len(pil_images)}"

        messages = []
        for _p, _img in zip(prompts, pil_images):
            q = self.QUESTION_TEMPLATE.format(Question=self.PROMPT)
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": _img},
                        {"type": "text", "text": q},
                    ],
                }
            ])

        all_scores: List[float] = []
        bs = self.batch_size
        for i in range(0, len(messages), bs):
            batch_messages = messages[i:i + bs]

            text = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)

            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                use_cache=True,
                do_sample=False,  # deterministic
                max_new_tokens=self.max_new_tokens,
            )
            gen_trim = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            out_text = self.processor.batch_decode(
                gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for t in out_text:
                s = self._parse_score(t)
                if s is None:
                    s = 3.0
                all_scores.append(s)

        scores = torch.tensor(all_scores, device=self.device, dtype=torch.float32)
        if return_raw:
            return scores
        return scores / 5.0 if self.normalize else scores


# ------------------- usage -------------------
def main():
    scorer = VisualQualityR1Scorer(
        model_path="TianheWu/VisualQuality-R1-7B",
        device="cuda",
        dtype=torch.bfloat16,
        batch_size=8,
        normalize=True,
        use_thinking=False,
        max_new_tokens=32,
    )

    # Case A: PIL list
    imgs_pil = [Image.open("test.png").convert("RGB")]
    prompts = ["unused"]
    print("PIL normalized:", scorer(prompts, imgs_pil))
    print("PIL raw:", scorer(prompts, imgs_pil, return_raw=True))

    # Case B: Tensor in [-1,1], [B,C,H,W]
    # x = torch.randn(1,3,512,512, device="cuda").clamp(-1,1)
    # print("Tensor normalized:", scorer(["unused"], x))

if __name__ == "__main__":
    main()

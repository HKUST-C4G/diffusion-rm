from typing import List, Optional
import torch
import torch.nn.functional as F

from t2v_metrics.models.vqascore_models.clip_t5_model import (
    CLIPT5Model,
    default_question_template,
    default_answer_template,
    format_question,
    format_answer,
)
from t2v_metrics.models.vqascore_models.mm_utils import (
    t5_tokenizer_image_token,
)
from t2v_metrics.constants import IGNORE_INDEX


def _parse_size(size):
    # size could be int or dict from HF image processor
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, dict):
        if "shortest_edge" in size:
            s = int(size["shortest_edge"])
            return (s, s)
        h = int(size.get("height", 224))
        w = int(size.get("width", 224))
        return (h, w)
    raise ValueError(f"Invalid size: {size}")


def _resize_shortest_edge_bchw(x: torch.Tensor, target: int) -> torch.Tensor:
    # x: [B,C,H,W] float
    B, C, H, W = x.shape
    short = min(H, W)
    if short == target:
        return x
    scale = target / float(short)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    return F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)


def _center_crop_bchw(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    B, C, H, W = x.shape
    if H < out_h or W < out_w:
        # should not happen if we resize shortest edge >= target
        pad_h = max(0, out_h - H)
        pad_w = max(0, out_w - W)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        B, C, H, W = x.shape
    top = (H - out_h) // 2
    left = (W - out_w) // 2
    return x[:, :, top:top + out_h, left:left + out_w]


def _pad_to_square_bchw(x: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W], bg: [1,3,1,1] in [0,1]
    B, C, H, W = x.shape
    S = max(H, W)
    out = bg.expand(B, C, S, S).clone()
    top = (S - H) // 2
    left = (S - W) // 2
    out[:, :, top:top + H, left:left + W] = x
    return out


class VQAScoreScorer(torch.nn.Module):
    """
    Tensor-only VQAScore (CLIP-FlanT5) scorer.

    Input:
      - prompts: List[str]
      - images:  torch.Tensor [B,3,H,W] in [-1, 1] (or [3,H,W])

    Output:
      - scores: torch.Tensor [B] float32, in (0,1], higher = better alignment
    """
    def __init__(
        self,
        model_name: str = "clip-flant5-xl",  # or "clip-flant5-xxl"
        device: str = "cuda",
        autocast_dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 8,
        pad_to_square: Optional[bool] = None,  # None => follow model config (image_aspect_ratio == "pad")
    ):
        super().__init__()
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.batch_size = batch_size

        # Load same core model as t2v_metrics.VQAScore
        self.model_wrapper = CLIPT5Model(model_name=model_name, device=device)
        self.tokenizer = self.model_wrapper.tokenizer
        self.model = self.model_wrapper.model
        self.image_processor = self.model_wrapper.image_processor

        self.conversational_style = getattr(self.model_wrapper, "conversational_style", "t5_chat")
        self.image_aspect_ratio = getattr(self.model_wrapper, "image_aspect_ratio", "pad")
        if pad_to_square is None:
            self.pad_to_square = (self.image_aspect_ratio == "pad")
        else:
            self.pad_to_square = bool(pad_to_square)

        # Pull preprocess config from processor
        cfg = self.image_processor.to_dict() if hasattr(self.image_processor, "to_dict") else {}
        self.do_resize = bool(cfg.get("do_resize", True))
        self.do_center_crop = bool(cfg.get("do_center_crop", True))
        self.do_normalize = bool(cfg.get("do_normalize", True))

        # target size / crop size
        self.resize_size = _parse_size(cfg.get("size", getattr(self.image_processor, "size", 224)))
        # For CLIP processors, crop is usually square 224
        crop_cfg = cfg.get("crop_size", getattr(self.image_processor, "crop_size", self.resize_size))
        self.crop_size = _parse_size(crop_cfg)

        # mean/std
        mean = torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,3,H,W] in [-1,1]
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expect images [B,3,H,W], got {tuple(images.shape)}")

        x = images.to(torch.float32)

        # [-1,1] -> [0,1]
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

        # optional pad-to-square using background = mean (already in [0,1])
        if self.pad_to_square:
            x = _pad_to_square_bchw(x, self._mean)

        # resize + center crop (match typical HF CLIP pipeline: resize shortest edge then center crop)
        if self.do_resize:
            # if resize_size is square, we mimic "shortest_edge"
            # many CLIP processors effectively do: shortest_edge -> 224
            target = min(self.resize_size)
            x = _resize_shortest_edge_bchw(x, target)

        if self.do_center_crop:
            ch, cw = self.crop_size
            x = _center_crop_bchw(x, ch, cw)

        if self.do_normalize:
            x = (x - self._mean) / self._std

        return x.to(self.device)

    def _build_qa(self, texts: List[str], question_template: str, answer_template: str):
        questions = [question_template.format(t) for t in texts]
        answers = [answer_template.format(t) for t in texts]
        questions = [format_question(q, conversation_style=self.conversational_style) for q in questions]
        answers = [format_answer(a, conversation_style=self.conversational_style) for a in answers]
        return questions, answers

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        prompts: List[str],
        question_template: str = default_question_template,
        answer_template: str = default_answer_template,
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list) or not all(isinstance(t, str) for t in prompts):
            raise TypeError("prompts must be List[str]")
        if not torch.is_tensor(images):
            raise TypeError("images must be torch.Tensor [B,3,H,W] in [-1,1]")

        if images.ndim == 3:
            images = images.unsqueeze(0)
        B = images.shape[0]
        if len(prompts) != B:
            raise ValueError(f"len(prompts)={len(prompts)} != batch size={B}")

        scores_all = []
        for i in range(0, B, self.batch_size):
            imgs = images[i:i + self.batch_size]
            txts = prompts[i:i + self.batch_size]

            pixel_values = self._preprocess_tensor(imgs)  # [b,3,224,224] normalized

            questions, answers = self._build_qa(txts, question_template, answer_template)

            input_ids = [t5_tokenizer_image_token(q, self.tokenizer, return_tensors="pt") for q in questions]
            labels = [t5_tokenizer_image_token(a, self.tokenizer, return_tensors="pt") for a in answers]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )

            input_ids = input_ids[:, : self.tokenizer.model_max_length].to(self.device)
            labels = labels[:, : self.tokenizer.model_max_length].to(self.device)

            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            decoder_attention_mask = labels.ne(IGNORE_INDEX)

            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                    images=pixel_values,
                    return_dict=True,
                )
                logits = out.logits

            # prob = exp(-CE), same spirit as original VQAScore implementation
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            probs = []
            for k in range(logits.shape[0]):
                probs.append(( -loss_fct(logits[k], labels[k]) ).exp())
            probs = torch.stack(probs, dim=0).to(torch.float32)  # [b]
            scores_all.append(probs)

        return torch.cat(scores_all, dim=0)


# ------------------- Usage -------------------
def main():
    scorer = VQAScoreScorer(
        model_name="clip-flant5-xl",
        device="cuda",
        autocast_dtype=torch.bfloat16,
        batch_size=8,
        pad_to_square=None,
    )

    # images: [-1,1], [B,3,H,W]
    images = torch.randn(2, 3, 512, 512, device="cuda").clamp(-1, 1)
    prompts = ["a cat sitting on a windowsill", "a red car on the road"]
    scores = scorer(prompts, images)
    print(scores)

if __name__ == "__main__":
    main()

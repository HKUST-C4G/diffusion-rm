from typing import List, Union, Optional
import torch
from PIL import Image
from torchvision import transforms
import pyiqa


class MANIQAScorer(torch.nn.Module):
    """
    pyiqa MANIQA scorer

    Input:
      - prompts: List[str] (kept for API compatibility, not used by MANIQA)
      - images:
          * List[PIL.Image]  OR
          * torch.Tensor [B,C,H,W] or [C,H,W], RGB, value in [0,1] recommended

    Output:
      - scores: torch.Tensor [B] (float32)
    """
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        model_name: str = "maniqa",   # or: "maniqa-kadid", "maniqa-pipal"
        enable_grad: bool = False,    # True => as_loss=True (allows backward)
        normalize_to_0_1: bool = False,  # min-max using metric.score_range when available
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.normalize_to_0_1 = normalize_to_0_1

        # create metric
        self.metric = pyiqa.create_metric(
            model_name,
            device=self.device,
            as_loss=enable_grad,  # docs: as_loss=True enables gradient propagation
        ).to(self.device)

        # metric output range (rough)
        self.score_range = getattr(self.metric, "score_range", None)

        # PIL -> tensor
        self.to_tensor = transforms.ToTensor()  # -> [0,1], RGB if PIL is RGB

    def _coerce_images(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            xs = []
            for im in images:
                if not isinstance(im, Image.Image):
                    raise TypeError(f"List must contain PIL.Image, got {type(im)}")
                xs.append(self.to_tensor(im.convert("RGB")))  # [3,H,W], float in [0,1]
            x = torch.stack(xs, dim=0)  # [B,3,H,W]

        elif torch.is_tensor(images):
            x = images
            if x.ndim == 3:
                x = x.unsqueeze(0)      # [1,C,H,W]
            if x.ndim != 4:
                raise ValueError(f"Tensor images must be [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
            if x.shape[1] not in (1, 3):
                raise ValueError(f"Expect C=1 or 3 in [B,C,H,W], got C={x.shape[1]}")

            # [-1,1] -> [0,1]
            x = x.to(torch.float32)
            x = (x + 1.0) * 0.5
            x = x.clamp(0.0, 1.0)

            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        return x.to(device=self.device, dtype=self.dtype)
    
    def _maybe_normalize(self, s: torch.Tensor) -> torch.Tensor:
        if not self.normalize_to_0_1:
            return s
        if not self.score_range or len(self.score_range) != 2:
            return s

        lo, hi = self.score_range
        # some metrics have "~" typical range; still useful as a clamp+scale
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
        if lo is None or hi is None or hi <= lo:
            return s
        return ((s - lo) / (hi - lo)).clamp(0, 1)

    def __call__(self, images: Union[List[Image.Image], torch.Tensor], prompts: List[str]=None) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        x = self._coerce_images(images)           # [B,3,H,W]
        scores = self.metric(x)                   # [B] or [B,1] depending on metric
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        scores = scores.to(torch.float32)
        scores = self._maybe_normalize(scores)
        return scores


# ------------------- Usage -------------------
def main():
    scorer = MANIQAScorer(
        device="cuda",
        dtype=torch.float32,
        model_name="maniqa-pipal",
        enable_grad=False,
        normalize_to_0_1=False,
    )

    imgs = [Image.open("test.png").convert("RGB")]
    prompts = ["unused"]
    print(scorer(prompts, imgs))   # tensor([..])

if __name__ == "__main__":
    main()

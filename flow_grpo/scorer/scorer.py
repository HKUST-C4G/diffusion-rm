from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import mindspore as ms
import numpy as np
from PIL import Image


class Scorer(ABC):

    @abstractmethod
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Return the scoring value of the images"""
        pass

    @staticmethod
    def array_to_images(images: Union[np.ndarray, ms.Tensor]) -> List[Image.Image]:
        if isinstance(images, ms.Tensor):
            images = images.transpose(0, 2, 3, 1).numpy()
        assert images.shape[-1] == 3, "must be in NHWC format"
        images = (images * 255).round().clip(0, 255).astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images

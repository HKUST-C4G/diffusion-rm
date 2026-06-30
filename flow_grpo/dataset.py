import os
from typing import Optional

import numpy as np
from mindspore.dataset import Sampler


class TextPromptDataset:

    def __init__(
        self, dataset: str, split: str = "train", max_num: Optional[int] = None
    ) -> None:
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]
        if max_num is not None:
            self.prompts = self.prompts[:max_num]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class DistributedKRepeatSampler(Sampler):

    def __init__(
        self,
        batch_size: int,
        k: int = 1,
        num_shards: Optional[int] = None,
        shard_id: Optional[int] = None,
        num_iters: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.k = k
        self.num_shards = num_shards if num_shards is not None else 1
        self.shard_id = shard_id if shard_id is not None else 0
        self.num_iters = num_iters
        if self.num_shards * self.batch_size % self.k != 0:
            raise ValueError(
                f"num_shards * batch_size ({self.num_shards * self.batch_size}) must be divisible by k ({self.k})"
            )
        self.sample_size = self.num_shards * self.batch_size // self.k

    def __iter__(self):
        num_iters = self.num_iters if self.num_iters is not None else self.dataset_size
        for _ in range(num_iters):
            sample_indices = np.random.choice(
                self.dataset_size, self.sample_size, replace=False
            )
            sample_indices = np.repeat(sample_indices, self.k)
            sample_indices = np.random.permutation(sample_indices)

            start = self.shard_id * self.batch_size
            end = start + self.batch_size
            yield sample_indices[start:end]

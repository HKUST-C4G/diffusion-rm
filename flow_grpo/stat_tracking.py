from collections import defaultdict
from typing import List

import numpy as np


class PerPromptStatTracker:

    def __init__(self, global_std: bool = False) -> None:
        self.global_std = global_std
        self.stats = defaultdict(list)

    def update(self, prompts: List[str], rewards: np.ndarray) -> np.ndarray:
        """
        prompts (N, )
        rewards (N, T)
        """
        prompts = np.array(prompts)
        unique_prompts = np.unique(prompts)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        for prompt in unique_prompts:
            prompt_rewards = rewards[prompts == prompt]
            self.stats[prompt].extend(prompt_rewards)

        for prompt, prompt_reward in self.stats.items():
            self.stats[prompt] = np.stack(prompt_reward)

        for prompt, prompt_reward in self.stats.items():
            mean = np.mean(prompt_reward, axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(prompt_reward, axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_reward - mean) / std
        return advantages

    def clear(self):
        self.stats = defaultdict(list)


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = np.array([1, 2, 3, 4, 5, 6])
    rewards = np.tile(rewards[..., None], (1, 10))
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)


if __name__ == "__main__":
    main()

import math

import mindspore as ms
import mindspore.mint as mint


def compute_log_prob(
    prev_sample: ms.Tensor, prev_sample_mean: ms.Tensor, var_t: ms.Tensor, dt: ms.Tensor
) -> ms.Tensor:
    log_prob = (
        -((prev_sample - prev_sample_mean) ** 2) / (-2 * var_t * dt)
        - mint.log(-1 * var_t * dt) / 2
        - math.log(2 * math.pi) / 2
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return log_prob

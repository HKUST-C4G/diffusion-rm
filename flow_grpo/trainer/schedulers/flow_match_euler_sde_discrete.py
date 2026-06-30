from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import numpy as np
from mindone.diffusers import FlowMatchEulerDiscreteScheduler
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.diffusers.utils.outputs import BaseOutput


@dataclass
class FlowMatchEulerSDEDiscreteSchedulerOutput(BaseOutput):
    prev_sample: ms.Tensor
    prev_sample_mean: Optional[ms.Tensor] = None
    var_t: Optional[ms.Tensor] = None
    dt: Optional[ms.Tensor] = None


class FlowMatchEulerSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """
    FlowMatch Euler SDE Discrete Scheduler.

    This class extends the FlowMatchEulerDiscreteScheduler to support SDE (Stochastic Differential Equations) sampling.
    Following https://arxiv.org/abs/2505.05470
    """

    def step(
        self,
        model_output: ms.Tensor,
        timestep: Union[float, ms.Tensor],
        sample: ms.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[np.random.Generator] = None,
        return_dict: bool = False,
    ) -> Union[FlowMatchEulerSDEDiscreteSchedulerOutput, Tuple]:
        if isinstance(timestep, int) or (
            isinstance(timestep, ms.Tensor)
            and timestep.dtype in (ms.int16, ms.int32, ms.int64)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(ms.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample_mean, var_t, dt = self.compute_prev_sample_mean(
            model_output, sample, sigma, sigma_next
        )

        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            dtype=model_output.dtype,
        )

        prev_sample = prev_sample_mean + mint.sqrt(-var_t * dt) * variance_noise

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, prev_sample_mean, var_t, dt)

        return FlowMatchEulerSDEDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            prev_sample_mean=prev_sample_mean,
            var_t=var_t,
            dt=dt,
        )

    def compute_prev_sample_mean(
        self,
        model_output: ms.Tensor,
        sample: ms.Tensor,
        sigma: ms.Tensor,
        sigma_next: ms.Tensor,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        sigma_max = self.sigmas[1]
        dt = sigma_next - sigma

        # 0.7 is the hyperparameter (a) used in the original paper
        var_t = (0.7**2) * sigma / (1 - mint.where(sigma == 1, sigma_max, sigma))

        coeff_0 = 1 + var_t / (2 * sigma) * dt
        coeff_1 = (1 + var_t * (1 - sigma) / (2 * sigma)) * dt

        prev_sample_mean = coeff_0 * sample + coeff_1 * model_output
        return prev_sample_mean, var_t, dt

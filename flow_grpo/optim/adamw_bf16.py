import mindspore as ms
import mindspore.mint as mint
from mindspore.mint.optim.adamw import (
    _run_optim_adamw_amsgrad_opt,
    _run_optim_adamw_opt,
)
from mindspore.ops import functional as F


def update_param(source: ms.Tensor, target: ms.Tensor) -> None:
    target.copy_(source.to(target.dtype))


class BF16AdamW(mint.optim.AdamW):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        maximize=False
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )

        # maintain a copy of parameters in fp32 for the optimiser update
        self.master_parameters = ms.ParameterTuple(
            [
                ms.Parameter(x.to(ms.float32), name="master." + x.name)
                for x in self.parameters
            ]
        )

    def construct(self, gradients):
        self.state_step.add_(self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            maximize = group.get("maximize")
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            lr = group.get("lr")
            grads = tuple(gradients[start_id:end_id])

            if group.get("amsgrad"):
                self.hyper_map(
                    F.partial(
                        _run_optim_adamw_amsgrad_opt,
                        self.adamw_opt,
                        beta1,
                        beta2,
                        float(lr),
                        group.get("eps"),
                        group.get("weight_decay"),
                        self.state_step,
                        group.get("amsgrad"),
                        maximize,
                    ),
                    self.master_parameters[start_id:end_id],
                    grads,
                    self.exp_avg[start_id:end_id],
                    self.exp_avg_sq[start_id:end_id],
                    group.get("max_exp_avg_sq"),
                )
            else:
                self.hyper_map(
                    F.partial(
                        _run_optim_adamw_opt,
                        self.adamw_opt,
                        beta1,
                        beta2,
                        float(lr),
                        group.get("eps"),
                        group.get("weight_decay"),
                        self.state_step,
                        group.get("amsgrad"),
                        maximize,
                    ),
                    self.master_parameters[start_id:end_id],
                    grads,
                    self.exp_avg[start_id:end_id],
                    self.exp_avg_sq[start_id:end_id],
                )

        self.hyper_map(update_param, self.master_parameters, self.parameters)
        return True

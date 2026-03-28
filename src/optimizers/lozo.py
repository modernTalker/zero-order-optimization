# optimizers/lozo.py
from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Union, Iterable, Dict, Any


class LOZO(ZeroOrderOptimizer):
    """
    Low-Rank Zeroth-Order (LOZO) optimizer.
    Matches the official implementation from LOZOtrainer.py.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        rank: int = 1,
        step_interval: int = 1,
        lozo_optimizer: str = "sgd",      # "sgd" or "sgdm"
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        tensor_sampling_type: str = "standard_normal",
        matrix_sampling_type: str = None,
        perturbation_mode: str = "two_side",   # not used by LOZO but kept for base compatibility
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            momentum=0.0,          # not used
            weight_decay=weight_decay,
            tensor_sampling_type=tensor_sampling_type,
            matrix_sampling_type=matrix_sampling_type,
            perturbation_mode=perturbation_mode,
        )

        self.rank = rank
        self.step_interval = step_interval
        self.lozo_optimizer = lozo_optimizer
        self.beta1 = beta1
        self.step = 0
        self.projected_grad = 0.0

        # Per-parameter state initialization
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if param.ndim >= 2:
                    state["v"] = None
                    state["v_old"] = None
                    state["exp_avg_m"] = None
                else:
                    state["exp_avg_m"] = None

    @torch.no_grad()
    def step(self, closure=None):
        """One LOZO step: two forward passes + update."""
        if closure is None:
            raise ValueError("LOZO requires a closure that returns the loss")

        # --- sample seed once per optimizer step ---
        self.zo_random_seed = np.random.randint(1_000_000_000)

        # First function evaluation (+ε perturbation)
        self._lowrank_zo_perturb_parameters(scaling_factor=1.0)
        loss1 = closure()

        # Second function evaluation (-2ε perturbation)
        self._lowrank_zo_perturb_parameters(scaling_factor=-2.0)
        loss2 = closure()

        self.projected_grad = ((loss1 - loss2) / (2 * self.defaults["eps"])).item()

        # Reset parameters to original values
        self._lowrank_zo_perturb_parameters(scaling_factor=1.0)

        # Parameter update
        if self.lozo_optimizer == "sgd":
            self._lowrank_zo_update()
        elif self.lozo_optimizer == "sgdm":
            self._lowrank_zo_update_momentum()
        else:
            raise ValueError(f"Unsupported lozo_optimizer: {self.lozo_optimizer}")

        # Advance step counter (matches official LOZOtrainer.py timing)
        self.step += 1
        return loss1

    def _lowrank_zo_perturb_parameters(self, scaling_factor: float = 1.0):
        """Low-rank perturbation: u v^T for matrices, standard Gaussian for vectors."""
        self.generator.manual_seed(self.zo_random_seed)

        for group in self.param_groups:
            eps = group["eps"]
            for param in group["params"]:
                state = self.state[param]
                if param.ndim >= 2:
                    # Resample v every step_interval steps
                    if self.step % self.step_interval == 0:
                        v = torch.randn(
                            param.size(1), self.rank,
                            device=param.device, dtype=param.dtype,
                            generator=self.generator
                        )
                        state["v"] = v
                    else:
                        v = state["v"]

                    # u is always freshly sampled (but reproducible via seed)
                    u = torch.randn(
                        param.size(0), self.rank,
                        device=param.device, dtype=param.dtype,
                        generator=self.generator
                    )
                    param.data.add_(scaling_factor * (u @ v.t()) * eps)
                else:
                    # Vector case: standard Gaussian
                    z = torch.randn(
                        param.size(), device=param.device, dtype=param.dtype,
                        generator=self.generator
                    )
                    param.data.add_(scaling_factor * z * eps)

    def _lowrank_zo_update(self):
        """Plain SGD update (no momentum)."""
        self.generator.manual_seed(self.zo_random_seed)
        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            for param in group["params"]:
                state = self.state[param]
                if param.ndim >= 2:
                    v = state["v"]
                    u = torch.randn(
                        param.size(0), self.rank,
                        device=param.device, dtype=param.dtype,
                        generator=self.generator
                    )
                    direction = u @ v.t()
                    # Matrices always get weight decay (matches original)
                    param.data.add_(-lr * (self.projected_grad * direction + wd * param.data))
                else:
                    z = torch.randn(
                        param.size(), device=param.device, dtype=param.dtype,
                        generator=self.generator
                    )
                    # 1D parameters (bias / LayerNorm) get NO weight decay
                    param.data.add_(-lr * (self.projected_grad * z))

    def _lowrank_zo_update_momentum(self):
        """SGD + low-rank momentum (sgdm)."""
        self.generator.manual_seed(self.zo_random_seed)
        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            for param in group["params"]:
                state = self.state[param]
                if param.ndim >= 2:
                    v = state["v"]
                    u = torch.randn(
                        param.size(0), self.rank,
                        device=param.device, dtype=param.dtype,
                        generator=self.generator
                    )

                    # Momentum logic (exact port from official LOZO)
                    if self.step % self.step_interval == 0:
                        if state.get("v_old") is not None:
                            v_old = state["v_old"]
                            n = v_old.shape[0]
                            if state.get("exp_avg_m") is None:
                                state["exp_avg_m"] = torch.zeros(
                                    (param.size(0), self.rank),
                                    device=param.device, dtype=param.dtype
                                )
                            tmp = (state["exp_avg_m"] @ v_old.t() @ v) / n
                            state["exp_avg_m"] = (
                                self.beta1 * tmp
                                + (1 - self.beta1) * self.projected_grad * u
                            )
                        else:
                            state["exp_avg_m"] = self.projected_grad * u
                    elif self.step % self.step_interval == self.step_interval - 1:
                        state["v_old"] = v.clone()
                        if state.get("exp_avg_m") is None:
                            state["exp_avg_m"] = torch.zeros(
                                (param.size(0), self.rank),
                                device=param.device, dtype=param.dtype
                            )
                        state["exp_avg_m"] = (
                            self.beta1 * state["exp_avg_m"]
                            + (1 - self.beta1) * self.projected_grad * u
                        )
                    else:
                        if state.get("exp_avg_m") is None:
                            state["exp_avg_m"] = torch.zeros(
                                (param.size(0), self.rank),
                                device=param.device, dtype=param.dtype
                            )
                        state["exp_avg_m"] = (
                            self.beta1 * state["exp_avg_m"]
                            + (1 - self.beta1) * self.projected_grad * u
                        )

                    direction = state["exp_avg_m"] @ v.t()
                    param.data.add_(-lr * (direction + wd * param.data))

                else:
                    # Vector case (momentum)
                    z = torch.randn(
                        param.size(), device=param.device, dtype=param.dtype,
                        generator=self.generator
                    )
                    if self.step == 0:
                        state["exp_avg_m"] = self.projected_grad * z
                    else:
                        if state.get("exp_avg_m") is None:
                            state["exp_avg_m"] = torch.zeros_like(z)
                        state["exp_avg_m"] = (
                            self.beta1 * state["exp_avg_m"]
                            + (1 - self.beta1) * self.projected_grad * z
                        )
                    # 1D parameters get NO weight decay
                    param.data.add_(-lr * state["exp_avg_m"])
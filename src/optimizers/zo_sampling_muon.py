from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple
from .opt_utils import *

class ZO_SamplingMUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            matrix_sampling_type: str = "Householder_reflection",
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side",
            device: str = "cuda" # FIXME: maybe change it
        ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity,
            vector_sampling_type=vector_sampling_type,
            device=device,
        )
        self.lr = lr
        self.perturbation_mode = perturbation_mode
        self.matrix_sampler = MatrixSampler(matrix_sampling_type, device=device)

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()  

        self.zo_random_seed = np.random.randint(1_000_000_000) 
        self.generator.manual_seed(self.zo_random_seed)
        
        shapes = [(name, tuple(param.shape)) for name, param in self.named_parameters_to_optim if param.ndim >= 2 and param.size(0) < 10000]
        E_dict = self.matrix_sampler.sample(shapes)

        if self._inner_optimizers is not None:
            for group_idx, _ in enumerate(self.param_groups):
                self._inner_optimizers[group_idx].zero_grad()
            original_grads = {}
            for name, param in self.named_parameters_to_optim:
                original_grads[name] = param.grad.clone() if param.grad is not None else None

        self.zo_perturb_parameters(scaling_factor=1, random_seed=self.zo_random_seed)
        self.generator.manual_seed(self.zo_random_seed)
        if closure is not None:
            with torch.enable_grad():
                loss1 = closure()

        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1, random_seed=self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                with torch.enable_grad():
                    loss2 = closure()
            self.projected_grad = torch.sign(loss2-loss1).item()
        else:  
            self.zo_perturb_parameters(scaling_factor=-2, random_seed=self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                with torch.enable_grad():
                    loss2 = closure()
            self.projected_grad = torch.sign(loss2-loss1).item()
            self.zo_perturb_parameters(scaling_factor=1, random_seed=self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
        
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                name = next(name for name, p in self.named_parameters_to_optim if p is param)
                device = param.device

                if param.ndim >= 2 and param.size(0) < 10000:
                    E, U, S, V = E_dict[name]
                    grad_update_final = self.projected_grad * (U @ V.T)
                else:
                    z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)

                    mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                    if mask is not None:
                        z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0
                    grad_update_final = self.projected_grad * z
                grad_update_final = grad_update_final.to(device)
                if self._inner_optimizers is None:
                    param.data.add_(grad_update_final, alpha=-self.lr) 
                else:
                    param.grad = grad_update_final
                    self._inner_optimizers[group_idx].step()

        if self._inner_optimizers is not None:
            for name, param in self.named_parameters_to_optim:
                param.grad = original_grads[name]

            for group_idx, _ in enumerate(self.param_groups):
                self._lr_schedulers[group_idx].step()

        return loss1

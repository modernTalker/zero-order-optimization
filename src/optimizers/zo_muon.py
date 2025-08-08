from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple

from .opt_utils import *

class ZO_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            vector_sampling_type: str = "standard_normal",
            perturbation_mode: str = "two_side"
        ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            vector_sampling_type=vector_sampling_type,
            gradient_sparsity=gradient_sparsity,
        )
        self.lr = lr 
        self.perturbation_mode = perturbation_mode

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()  

        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)

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
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="one_side")
        else:  
            self.zo_perturb_parameters(scaling_factor=-2, random_seed=self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
            if closure is not None:
                with torch.enable_grad():
                    loss2 = closure()
            self.projected_grad = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")
            self.zo_perturb_parameters(scaling_factor=1, random_seed=self.zo_random_seed)
            self.generator.manual_seed(self.zo_random_seed)
        
        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                name = next(name for name, p in self.named_parameters_to_optim if p is param)
                device = param.device

                z = self.vector_sampler.sample(param.shape, generator=self.generator).to(device)
                self.generator.manual_seed(self.zo_random_seed)
                grad_sparsity = self.get_grad_sparsity_by_name(name) 
                mask = getattr(self, 'get_grad_sparsity_by_name', lambda x: None)(name)
                if mask is not None:
                    z[fast_random_mask_like(z, mask, generator=self.sparse_grad_rng)] = 0      
                grad_update = self.projected_grad * z

                if param.ndim >= 2 and param.size(0) < 10000:
                    g = grad_update.to(torch.bfloat16)
                    original_shape = g.shape
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)  
                    g_ortho = zeropower_via_newtonschulz5(g, steps=5)
                    if len(original_shape) > 2:
                        g_ortho = g_ortho.view(original_shape)
                    grad_update_final = g_ortho.to(param.data.dtype)
                else:
                    grad_update_final = grad_update

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

from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable
import time

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            beta: float = 0.9,
            use_smoothing: bool = True,
            lr: float = 0.01,
            eps: float = 1e-3,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
            q: int = 1,
            module_wise_perturbation: bool = False,
            coordinate_perturbation: bool = False
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            momentum=momentum,
            beta=beta,
            use_smoothing=use_smoothing,
            gradient_sparsity=gradient_sparsity
        )
        super().__init__(params, defaults)
        
        self.lr = lr 
        self.beta = beta
        self.use_smoothing = use_smoothing
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None 
        self._prepare_parameters()  

        for group in self.param_groups:
            for param in group['params']:    
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )
                state['step'] += 1

        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)

        self.indices_perturb(scaling_factor = 1.0)
        if closure is not None:
            with torch.enable_grad():
                loss1 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self.indices_perturb(scaling_factor = -2.0)
        if closure is not None:
            with torch.enable_grad():
                loss2 = closure()
        self.generator.manual_seed(self.zo_random_seed)

        self.indices_perturb(scaling_factor = 1.0)
        self.generator.manual_seed(self.zo_random_seed)

        grad_update = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode="two_side")

        for group in self.param_groups:
            for param in group['params']:
                if not any(name for name, p in self.named_parameters_to_optim if p is param):
                    continue
                state = self.state[param]
                indices = self.select_indices(param_shape=param.shape, device=param.device)
                
                if self.use_smoothing:
                    if isinstance(indices, torch.Tensor):
                        state['grad_accum'][indices] = (
                            self.beta * state['grad_accum'][indices] + 
                            (1 - self.beta) * grad_update
                        )
                    else:
                        rows, cols = indices
                        state['grad_accum'][rows[:, None], cols] = (
                            self.beta * state['grad_accum'][rows[:, None], cols] + 
                            (1 - self.beta) * grad_update
                        )
                else:
                    if isinstance(indices, torch.Tensor):
                        state['grad_accum'][indices] = grad_update
                    else:
                        rows, cols = indices
                        state['grad_accum'][rows[:, None], cols] = grad_update
                
                update_direction = torch.sign(state['grad_accum'])
                param.data.add_(update_direction, alpha=-self.lr)

        return loss1
    
    def select_indices(self, param_shape, rows_ratio = 0.1, cols_ratio = 0.1, device='cuda'):

        if len(param_shape) == 1:
            n_elems = param_shape[0]
            k = max(1, int(n_elems * rows_ratio))
            indices = torch.randperm(n_elems, device=device, generator=self.generator)[:k]
            return indices
        
        n_rows, n_cols = param_shape
        k = max(1, int(n_rows * rows_ratio))
        m = max(1, int(n_cols * cols_ratio))

        selected_rows = torch.randperm(n_rows, device=device, generator=self.generator)[:k]
        selected_cols = torch.randperm(n_cols, device=device, generator=self.generator)[:m]
        return (selected_rows, selected_cols)
    
    def indices_perturb(self, scaling_factor = 1.0):
        for name, param in self.named_parameters_to_optim:
            indices = self.select_indices(param_shape=param.shape, device=param.device)
            if isinstance(indices, torch.Tensor):
                param.data[indices] += scaling_factor * self.zo_eps
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] += scaling_factor * self.zo_eps


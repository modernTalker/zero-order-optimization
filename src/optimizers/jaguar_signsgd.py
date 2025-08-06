from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Iterable

from .opt_utils import *

class Jaguar_SignSGD(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            tau: float = 0.01,
            beta: float = 0.9,
            use_smoothing: bool = True,
            lr: float = 0.01,
            eps: float = 1e-8,
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
            tau=tau,
            beta=beta,
            use_smoothing=use_smoothing
        )
        super().__init__(params, defaults)
        
        self.tau = tau
        self.beta = beta
        self.use_smoothing = use_smoothing
        self.perturbation_mode = perturbation_mode
        self.q = q
        self.module_wise_perturbation = module_wise_perturbation
        self.coordinate_perturbation = coordinate_perturbation

    @torch.no_grad()
    def step(self, closure):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(
                        param, 
                        memory_format=torch.preserve_format
                    )
                state['step'] += 1

        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        selected_indices = {}
        original_values = {}

        for name, param in self.named_parameters_to_optim:
            if len(param.shape) == 1:
                n_elements = param.numel()
                k = max(1, int(n_elements * 0.1))
                indices = torch.randperm(n_elements, device=param.device)[:k]
                selected_indices[name] = indices
                original_values[name] = param.data[indices].clone()
            else:
                n_rows, n_cols = param.shape
                k = max(1, int(n_rows * 0.1))
                m = max(1, int(n_cols * 0.1))
                selected_rows = torch.randperm(n_rows, device=param.device)[:k]
                selected_cols = torch.randperm(n_cols, device=param.device)[:m]
                selected_indices[name] = (selected_rows, selected_cols)
                original_values[name] = param.data[selected_rows[:, None], selected_cols].clone()

        for name, param in self.named_parameters_to_optim:
            indices = selected_indices[name]
            if isinstance(indices, torch.Tensor):
                param.data[indices] += self.tau
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] += self.tau
                
        loss1 = closure()

        for name, param in self.named_parameters_to_optim:
            indices = selected_indices[name]
            if isinstance(indices, torch.Tensor):
                param.data[indices] = original_values[name] - self.tau
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] = original_values[name] - self.tau
                
        loss2 = closure()

        for name, param in self.named_parameters_to_optim:
            indices = selected_indices[name]
            if isinstance(indices, torch.Tensor):
                param.data[indices] = original_values[name]
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] = original_values[name]

        grad_update = (loss1 - loss2).item() / (2 * self.tau)

        for group in self.param_groups:
            for param in group['params']:
                name = next(name for name, p in self.named_parameters_to_optim if p is param)
                state = self.state[param]
                indices = selected_indices[name]
                
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
                param.data.add_(update_direction, alpha=-group['lr'])

        return loss1

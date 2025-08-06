from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple

from .opt_utils import *

class Jaguar_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            tau: Optional[float] = None,
            beta: Optional[float] = None, 
            use_smoothing: Optional[bool] = None,
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            momentum: float = 0.0,
            gradient_sparsity: Optional[Union[float, Dict[str, float]]] = None,
            perturbation_mode: str = "two_side",
        ):
        super().__init__(
            params=params,
            lr=lr,
            eps=eps,
            momentum=momentum,
            gradient_sparsity=gradient_sparsity
        )
        self.tau = tau 
        self.beta = beta
        self.use_smoothing = use_smoothing
        self.lr = lr 

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None
        tau = self.tau
        beta = self.beta
        use_smoothing = self.use_smoothing

        self._prepare_parameters()   
                
        self.zo_random_seed = np.random.randint(1_000_000_000)
        torch.manual_seed(self.zo_random_seed)

        selected_indices = {}
        original_values = {}

        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(param.shape) == 0:  
                    continue
                if 'grad_accum' not in state:
                    state['grad_accum'] = torch.zeros_like(param.data)
                
                if len(param.data.shape) == 1:
                    n_elements = param.data.shape[0]
                    k = max(1, int(n_elements * 0.1))  
                    indices = torch.randperm(n_elements, device=param.device)[:k]
                    selected_indices[param] = indices
                    original_values[param] = param.data[indices].clone()
                else:
                    n_rows, n_cols = param.data.shape
                    k = max(1, int(n_rows * 0.1))
                    m = max(1, int(n_cols * 0.1))
                    selected_rows = torch.randperm(n_rows, device=param.device)[:k]
                    selected_cols = torch.randperm(n_cols, device=param.device)[:m]
                    selected_indices[param] = (selected_rows, selected_cols)
                    original_values[param] = param.data[selected_rows[:, None], selected_cols].clone()

        for param, indices in selected_indices.items():
            if len(param.data.shape) == 1:
                param.data[indices] += tau
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] += tau
                
        if closure is not None:
            with torch.enable_grad():
                loss1 = closure()
                
        for param, indices in selected_indices.items():
            if len(param.data.shape) == 1:
                param.data[indices] = original_values[param] - tau
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] = original_values[param] - tau
                
        if closure is not None:
            with torch.enable_grad():
                loss2 = closure()
                
        for param, indices in selected_indices.items():
            if len(param.data.shape) == 1:
                param.data[indices] = original_values[param]
            else:
                rows, cols = indices
                param.data[rows[:, None], cols] = original_values[param]

        grad_update = (loss1 - loss2).item() / (2 * tau) 

        for group_idx, group in enumerate(self.param_groups):
            for param in group['params']:
                if param not in selected_indices:
                    continue
                    
                state = self.state[param]
            
                if len(state) == 0:  
                    state['step'] = 0
                    state['grad_accum'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    
                indices = selected_indices[param]
                
                if use_smoothing:
                    if len(param.data.shape) == 1:
                        state['grad_accum'][indices] = beta * state['grad_accum'][indices] + (1 - beta) * grad_update
                    else:
                        rows, cols = indices
                        state['grad_accum'][rows[:, None], cols] = beta * state['grad_accum'][rows[:, None], cols] + (1 - beta) * grad_update
                else:
                    if len(param.data.shape) == 1:
                        state['grad_accum'][indices] = grad_update
                    else:
                        rows, cols = indices
                        state['grad_accum'][rows[:, None], cols] = grad_update
                
                if len(param.data.shape) == 1:
                    ns_accum = state['grad_accum'].clone()
                    ns_accum[indices] = torch.sign(ns_accum[indices])
                else:
                    ns_accum = zeropower_via_newtonschulz5(state['grad_accum'], steps=5).to(param.data.dtype)
                
                param.data.add_(ns_accum, alpha=-self.lr)

        return loss1

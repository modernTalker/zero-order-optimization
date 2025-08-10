from .base import ZeroOrderOptimizer
import torch
import numpy as np
from typing import Optional, Callable, Dict, Any, Union, List, Iterable, Tuple

from .opt_utils import *

class Jaguar_MUON(ZeroOrderOptimizer):
    def __init__(self, 
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
            # tau: Optional[float] = None,
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
            gradient_sparsity=gradient_sparsity,
            perturbation_mode=perturbation_mode
        )
        # self.tau = tau 
        self.beta = beta
        self.use_smoothing = use_smoothing
        self.lr = lr 

    @torch.no_grad()
    def step(self, closure=None):
        loss1, loss2 = None, None
        # tau = self.tau
        beta = self.beta
        use_smoothing = self.use_smoothing

        self._prepare_parameters()   
                
        self.zo_random_seed = np.random.randint(1_000_000_000)
        self.generator.manual_seed(self.zo_random_seed)

        selected_indices = self._select_perturbation_indices(row_frac=0.1, col_frac=0.1, min_elements=1)

        self.generator.manual_seed(self.zo_random_seed)

        self._apply_sparse_perturbation(selected_indices, scaling_factor=1)
        loss1 = closure()

        self._apply_sparse_perturbation(selected_indices, scaling_factor=-2)
        loss2 = closure()

        self._apply_sparse_perturbation(selected_indices, scaling_factor=1)

        grad_update = self.grad_approx(loss_original=loss1, loss_perturbed=loss2, perturbation_mode=self.perturbation_mode)

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
